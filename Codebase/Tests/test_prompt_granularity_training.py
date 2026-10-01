from __future__ import annotations

import copy
import json
import math
import shutil
import sys
import tempfile
import unittest
from functools import partial
from pathlib import Path
from unittest.mock import patch

CODEBASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODEBASE / "Core"))

import torch
from PIL import Image as PILImage
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sam3.model.model_misc import SAM3Output
from sam3.train.data.coco_json_loaders import COCO_FROM_JSON
from sam3.train.data.sam3_image_dataset import Datapoint, FindQueryLoaded, Image, InferenceMetadata, Object, Sam3ImageDataset
from sam3.train.loss.loss_fns import IABCEMdetr, instance_masks_to_semantic_masks
from sam3.train.loss.sam3_loss import Sam3LossWrapper
from sam3.train.utils.train_utils import collect_dict_keys

import config
from prompt_granularity import CATEGORIES, RUN_MANIFEST, verify_checkpoint_contract
from prompt_granularity_training import PromptGranularityLoss, collate_prompt_granularity


class ScalarLoss(torch.nn.Module):
    def forward(self, outputs, **kwargs):
        return {"core_loss": outputs["value"]}


def sample(empty=False):
    canopy = torch.zeros(8, 8, dtype=torch.bool)
    tree_a = torch.zeros(8, 8, dtype=torch.bool)
    tree_b = torch.zeros(8, 8, dtype=torch.bool)
    canopy[:5, :5] = True
    tree_a[2:7, 2:7] = True
    tree_b[6:, :2] = True
    masks = (canopy, tree_a, tree_b)
    boxes = ([0, 0, 5, 5], [2, 2, 7, 7], [0, 6, 2, 8])
    objects = [
        Object(bbox=torch.tensor(box, dtype=torch.float32), area=float(mask.sum()), object_id=index, segment=mask)
        for index, (box, mask) in enumerate(zip(boxes, masks))
    ]
    queries = []
    for category, text, object_ids in ((1, "tree canopy", [0]), (2, "tree", [1, 2])):
        metadata = InferenceMetadata(
            coco_image_id=7,
            original_image_id=7,
            original_category_id=category,
            original_size=(8, 8),
            object_id=-1,
            frame_index=0,
        )
        queries.append(
            FindQueryLoaded(
                query_text=text,
                image_id=0,
                object_ids_output=[] if empty else object_ids,
                is_exhaustive=True,
                inference_metadata=metadata,
            )
        )
    return Datapoint(find_queries=queries, images=[Image(data=torch.zeros(3, 8, 8), objects=objects, size=(8, 8))])


def annotation(identifier, image_id, category, bbox, *, crowd=0):
    x, y, width, height = bbox
    return {
        "id": identifier,
        "image_id": image_id,
        "category_id": category,
        "original_category_id": category,
        "bbox": list(bbox),
        "segmentation": [[x, y, x + width, y, x + width, y + height, x, y + height]],
        "area": width * height,
        "iscrowd": crowd,
    }


class TrainingTests(unittest.TestCase):
    def test_native_loss_is_scaled_by_images_without_detaching(self):
        for prompts in (1, 2):
            with self.subTest(prompts=prompts):
                value = torch.tensor(3.0, requires_grad=True)
                stages = SAM3Output([[{"value": value, "indices": ([], [], None)}]])
                targets = [{"num_boxes": torch.tensor([1, 0, 2, 0])}]
                kwargs = dict(loss_fns_find=[ScalarLoss()], normalization="local", scale_by_find_batch_size=True)
                native = Sam3LossWrapper(**kwargs)(stages, targets)["core_loss"]
                result = PromptGranularityLoss(queries_per_image=prompts, **kwargs)(stages, targets)
                torch.testing.assert_close(result["core_loss"], native / math.sqrt(prompts))
                result["core_loss"].backward()
                torch.testing.assert_close(value.grad, torch.tensor(math.sqrt(4 / prompts)))
                self.assertEqual(result["pg_image_count"].item(), 4 / prompts)

    def test_complete_paired_batches_semantic_unions_and_negative_queries(self):
        for empty in (False, True):
            data = collate_prompt_granularity([sample(empty), sample(empty)], "all", 2, with_seg_masks=True)["all"]
            self.assertEqual(data.find_text_batch, ["tree canopy", "tree"])
            self.assertEqual(data.find_inputs[0].img_ids.tolist(), [0, 0, 1, 1])
            self.assertEqual(data.find_targets[0].num_boxes.tolist(), [0, 0, 0, 0] if empty else [1, 2, 1, 2])
            if not empty:
                semantic = instance_masks_to_semantic_masks(data.find_targets[0].segments, data.find_targets[0].num_boxes)
                expected_canopy = sample().images[0].objects[0].segment
                expected_tree = sample().images[0].objects[1].segment | sample().images[0].objects[2].segment
                torch.testing.assert_close(semantic[0], expected_canopy)
                torch.testing.assert_close(semantic[1], expected_tree)
                self.assertTrue((semantic[0] & semantic[1]).any())
                self.assertFalse(torch.equal(semantic[0], semantic[1]))

    def test_wrong_query_order_or_missing_negative_query_is_rejected(self):
        for bad in ("order", "missing", "text"):
            value = sample()
            if bad == "order":
                value.find_queries.reverse()
            elif bad == "missing":
                value.find_queries.pop()
            else:
                value.find_queries[0].query_text = "tree"
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                collate_prompt_granularity([value], "all", 2, with_seg_masks=True)

    def test_validation_chunk_uses_one_query_per_image(self):
        values = []
        for index in (0, 1):
            value = sample()
            value.find_queries = [value.find_queries[index]]
            values.append(value)
        data = collate_prompt_granularity(values, "oamtcd", 1, with_seg_masks=True)["oamtcd"]
        self.assertEqual(data.find_inputs[0].img_ids.tolist(), [0, 1])
        self.assertEqual(data.find_targets[0].num_boxes.tolist(), [1, 2])
        self.assertEqual(data.find_metadatas[0].original_category_id.tolist(), [1, 2])

    def test_prompt_val_collator_key_matches_native_meter_discovery(self):
        with patch.object(config, "PROMPT_GRANULARITY_ENABLED", True):
            cfg = config.build_sam3_train_config(config.get_training_preset("full_ft"))
        val_config = OmegaConf.create(cfg["trainer"]["data"]["val"])
        val_keys = collect_dict_keys(val_config)
        meter_keys = list(cfg["trainer"]["meters"]["val"])
        self.assertEqual(val_keys, ["oamtcd"])
        self.assertEqual(set(val_keys), set(meter_keys))
        self.assertIn("collate_fn", cfg["trainer"]["data"]["val"]["collate_fn"]["_target_"])

    def test_native_loader_transforms_crowds_absence_and_crop_empty_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images/train"
            image_dir.mkdir(parents=True)
            images = []
            for image_id in range(3):
                filename = f"images/train/{image_id}.png"
                PILImage.new("RGB", (1152, 1152), color=(20, 40, 60)).save(root / filename)
                images.append({"id": image_id, "file_name": filename, "width": 1152, "height": 1152})
            annotations = [
                annotation(1, 0, 1, [0, 0, 1151, 1151]),
                annotation(2, 0, 2, [100, 100, 300, 300]),
                annotation(3, 0, 2, [500, 500, 250, 250]),
                annotation(4, 0, 2, [700, 100, 100, 100], crowd=1),
                annotation(5, 1, 1, [0, 0, 1151, 1151]),
                annotation(6, 2, 1, [0, 0, 1151, 1151]),
                annotation(7, 2, 2, [1000, 1000, 100, 100]),
            ]
            ann_path = root / "annotations.json"
            ann_path.write_text(json.dumps({"images": images, "annotations": annotations, "categories": CATEGORIES}))
            transforms = instantiate(config._train_transforms(), _convert_="all")
            dataset = Sam3ImageDataset(
                img_folder=str(root),
                ann_file=str(ann_path),
                transforms=transforms,
                max_ann_per_img=1000,
                multiplier=1,
                training=True,
                load_segmentation=True,
                use_caching=False,
                max_train_queries=10,
                max_val_queries=10,
                coco_json_loader=partial(COCO_FROM_JSON, include_negatives=True, category_chunk_size=2),
            )
            with patch("sam3.train.transforms.basic_for_api.T.RandomCrop.get_params", return_value=(0, 0, 896, 896)):
                mixed = dataset[0]
                absent = dataset[1]
                crop_empty = dataset[2]
            self.assertEqual([query.query_text for query in mixed.find_queries], ["tree canopy", "tree"])
            self.assertEqual([query.inference_metadata.original_category_id for query in mixed.find_queries], [1, 2])
            self.assertEqual([len(query.object_ids_output) for query in mixed.find_queries], [1, 2])
            self.assertEqual([len(query.object_ids_output) for query in absent.find_queries], [1, 0])
            self.assertEqual([len(query.object_ids_output) for query in crop_empty.find_queries], [1, 0])
            data = collate_prompt_granularity([mixed], "all", 2, with_seg_masks=True)["all"]
            semantic = instance_masks_to_semantic_masks(data.find_targets[0].segments, data.find_targets[0].num_boxes)
            split = torch.split(data.find_targets[0].segments, data.find_targets[0].num_boxes.tolist())
            boxes = torch.split(data.find_targets[0].boxes.reshape(-1, 4), data.find_targets[0].num_boxes.tolist())
            torch.testing.assert_close(semantic[0], torch.any(split[0], dim=0))
            torch.testing.assert_close(semantic[1], torch.any(split[1], dim=0))
            self.assertFalse(torch.equal(semantic[0], semantic[1]))
            self.assertEqual([len(value) for value in boxes], [1, 2])
            self.assertGreater(boxes[0][0, 2:].prod().item(), boxes[1][:, 2:].prod(dim=1).max().item())
            self.assertFalse(torch.equal(boxes[1][0], boxes[1][1]))
            crop_data = collate_prompt_granularity([crop_empty], "all", 2, with_seg_masks=True)["all"]
            crop_semantic = instance_masks_to_semantic_masks(crop_data.find_targets[0].segments, crop_data.find_targets[0].num_boxes)
            self.assertGreater(crop_semantic[0].sum().item(), 0)
            self.assertEqual(crop_semantic[1].sum().item(), 0)

    def test_absent_concept_keeps_presence_loss_but_suppresses_token_ce(self):
        self.assertTrue(torch.cuda.is_available(), "Native Triton presence loss requires CUDA.")
        device = torch.device("cuda")
        criterion = IABCEMdetr(
            pos_weight=10.0,
            weight_dict={"loss_ce": 1.0, "presence_loss": 1.0},
            gamma=2,
            weak_loss=False,
            alpha=0.25,
            use_presence=True,
        ).to(device)
        logits = torch.zeros(2, 2, 1, device=device, requires_grad=True)
        presence = torch.zeros(2, 1, device=device, requires_grad=True)
        outputs = {
            "pred_logits": logits,
            "pred_boxes_xyxy": torch.tensor([[[0.0, 0.0, 1.0, 1.0]] * 2] * 2, device=device),
            "presence_logit_dec": presence,
        }
        targets = {
            "boxes_xyxy": torch.tensor([[0.0, 0.0, 1.0, 1.0]], device=device),
            "boxes_padded": torch.tensor([[[0.5, 0.5, 1.0, 1.0]], [[0.0, 0.0, 0.0, 0.0]]], device=device),
            "object_ids_padded": torch.tensor([[0], [-1]], device=device),
            "is_exhaustive": torch.tensor([True, True], device=device),
        }
        indices = tuple(torch.tensor([0], device=device) for _ in range(3))
        losses = criterion(outputs=outputs, targets=targets, indices=indices, num_boxes=torch.tensor(1.0, device=device))
        losses["loss_ce"].backward(retain_graph=True)
        self.assertGreater(logits.grad[0].abs().sum().item(), 0)
        self.assertEqual(logits.grad[1].abs().sum().item(), 0)
        losses["presence_loss"].backward()
        self.assertGreater(presence.grad[1].abs().sum().item(), 0)

    def test_checkpoint_binding_survives_save_stage_reload_and_a0_passthrough(self):
        from progress_trainer import ProgressBarTrainer
        from sam3.train.trainer import Trainer

        instance = object.__new__(ProgressBarTrainer)
        checkpoint = {"model": {"weight": torch.zeros(1)}, "optimizer": {}, "loss": {}, "scaler": {}, "epoch": 1, "steps": 2}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {"purpose": "full", "contract": {"fixture": True}, "efficacy": {"fixture": True}}
            (root / RUN_MANIFEST).write_text(json.dumps(manifest))
            path = root / "checkpoints/checkpoint.pt"
            path.parent.mkdir()

            def save(payload, target):
                torch.save(payload, target)

            with patch.object(config, "PROMPT_GRANULARITY_ENABLED", True), patch.object(Trainer, "_save_checkpoint", side_effect=save):
                instance._save_checkpoint(checkpoint, path)
            staged = root / "staged.pt"
            shutil.copy2(path, staged)
            loaded = torch.load(staged, map_location="cpu", weights_only=True)
            verify_checkpoint_contract(loaded, manifest)
            self.assertNotIn("prompt_granularity", checkpoint)
            with patch.object(config, "PROMPT_GRANULARITY_ENABLED", False), patch.object(Trainer, "_save_checkpoint") as native:
                instance._save_checkpoint(checkpoint, path)
                self.assertIs(native.call_args.args[0], checkpoint)

    def test_incomplete_loss_pair_is_rejected(self):
        loss = PromptGranularityLoss(queries_per_image=2, loss_fns_find=[], scale_by_find_batch_size=True)
        with self.assertRaises(ValueError):
            loss(None, [{"num_boxes": torch.tensor([1, 2, 3])}])


if __name__ == "__main__":
    unittest.main(verbosity=2)
