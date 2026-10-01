from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import types
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

CODEBASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODEBASE / "Core"))
import prompt_granularity as pg
from prompt_granularity import (
    CATEGORIES, SMOKE_REPORT_VERSION, derive_view, digest, file_digest, prefix_identity,
    prepare_views, require_smoke, validate_efficacy_spec, validate_views,
    verify_checkpoint_contract,
)


def source_coco(split):
    return {
        "info": {"description": "fixture"},
        "images": [{"id": 7, "file_name": f"images/{split}/tile.png", "width": 2048, "height": 2048, "source_split": split}],
        "categories": [{"id": 1, "name": "tree"}],
        "annotations": [
            {"id": i, "image_id": 7, "category_id": 1, "original_category_id": cat,
             "bbox": [1, 2, 3, 4], "segmentation": [[1, 2, 4, 2, 4, 6]], "area": 12, "iscrowd": crowd}
            for i, cat, crowd in ((10, 1, 0), (11, 2, 0), (12, 2, 1))
        ],
    }


def smoke_fixture(root, contract):
    output = (root / "sam3_prompt_granularity_smoke_fixture").resolve()
    output.mkdir()
    checkpoint = output / "smoke_checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint fixture")
    fixtures = []
    for case, counts, per_micro in (
        ("mixed", {"tree": 8, "tree_canopy": 8}, (2, 2)),
        ("tree_only", {"tree": 8, "tree_canopy": 0}, (2, 0)),
        ("canopy_only", {"tree": 0, "tree_canopy": 8}, (0, 2)),
        ("empty", {"tree": 0, "tree_canopy": 0}, (0, 0)),
    ):
        fixtures.append({
            "case": case, "microbatches": 4, "losses": [1.0] * 4,
            "telemetry": [{"pg_pair_count": 4, "pg_image_count": 2, "pg_scale": 2 ** -0.5, "pg_tree_targets": per_micro[0], "pg_canopy_targets": per_micro[1]}] * 4,
            "target_counts": counts, "grad_norm": 1.0, "mask_grad": 0.0 if case == "empty" else 1.0,
            "optimizer_step_applied": True,
            "ddp_gradient_equal": True,
            "explicit_gradient_sync": True,
            "output_shapes": {"pred_masks": [4, 200, 288, 288], "pred_logits": [4, 200, 1], "pred_masks_o2m": [4, 200, 288, 288], "pred_logits_o2m": [4, 200, 1]},
        })
    ranks = []
    for rank in (0, 1):
        ranks.append({
            "rank": rank, "local_rank": rank, "world_size": 2,
            "device": "NVIDIA A40", "amp_dtype": "bfloat16", "micro_batch": 2,
            "accumulation_steps": 4, "ddp_accumulation_sync": "every_microbatch_plus_explicit_final_when_static_graph", "effective_image_batch": 16,
            "optimizer_steps": 5, "skipped_optimizer_steps": 0,
            "architecture": {"num_queries": 200, "num_layers": 6, "dac": True, "o2m_mask_predict": True, "aux_masks": False},
            "sources": {"core": str(CODEBASE / "Core/training_pipeline.py"), "sam3": str(CODEBASE / "Support/sam3/sam3/__init__.py")},
            "fixtures": copy.deepcopy(fixtures),
            "production_integration": {"status": "passed", "train_workers": 4, "val_workers": 2, "train_microbatches": 4, "optimizer_step_applied": True, "ddp_gradient_equal": True, "explicit_gradient_sync": True, "target_counts": {"tree": 1, "tree_canopy": 1}, "grad_norm": 1.0, "mask_grad": 1.0, "output_shapes": {"pred_masks": [4, 200, 288, 288], "pred_logits": [4, 200, 1], "pred_masks_o2m": [4, 200, 288, 288], "pred_logits_o2m": [4, 200, 1]}},
            "validation": {"status": "passed", "dummy_loss": True, "postprocessing": "passed", "dense_forward": "passed", "query_categories": [rank + 1], "prediction_categories": [rank + 1], "checkpoint_reload_equal": True, "checkpoint_fresh_model": True, "checkpoint_shared_inputs": True, "checkpoint_optimizer_steps": 5},
            "memory_mib": {"peak_allocated": 1024.0, "peak_reserved": 2048.0},
        })
    report = {
        "report_version": SMOKE_REPORT_VERSION, "status": "passed",
        "completed_utc": "2026-09-10T00:00:00+00:00", "output_dir": str(output),
        "world_size": 2, "skipped_checks": [], "contract": contract,
        "contract_sha256": digest(contract),
        "checkpoint": {"status": "passed", "reload_equal": True, "purpose": "smoke", "contract_sha256": digest(contract), "state_keys": ["contract", "loss", "model", "optimizer", "purpose", "scaler"], **prefix_identity(checkpoint)},
        "ranks": ranks,
    }
    path = output / "smoke_report.json"
    path.write_text(json.dumps(report))
    return path, report


class ViewTests(unittest.TestCase):
    def make_sources(self, root):
        (root / "annotations").mkdir()
        for split in ("train", "test"):
            (root / "annotations" / f"{split}_annotations.coco.json").write_text(json.dumps(source_coco(split)))

    def test_only_categories_and_policy_change(self):
        source = source_coco("train")
        before = copy.deepcopy(source)
        view = derive_view(source, "train", strict=False)
        self.assertEqual(source, before)
        self.assertEqual(view["categories"], CATEGORIES)
        self.assertEqual(view["images"], source["images"])
        for original, derived in zip(source["annotations"], view["annotations"]):
            expected = {**original, "category_id": original["original_category_id"]}
            self.assertEqual(derived, expected)
        view["annotations"][0]["bbox"][0] = 99
        self.assertEqual(source["annotations"][0]["bbox"][0], 1)

    def test_bad_original_categories_fail_closed(self):
        for bad in (None, 0, 3, "tree"):
            source = source_coco("train")
            if bad is None:
                source["annotations"][0].pop("original_category_id")
            else:
                source["annotations"][0]["original_category_id"] = bad
            with self.subTest(category=bad), self.assertRaises(ValueError):
                derive_view(source, "train", strict=False)

    def test_noncollapsed_input_is_rejected(self):
        source = source_coco("train")
        source["categories"] = CATEGORIES
        with self.assertRaises(ValueError):
            derive_view(source, "train", strict=False)

    def test_duplicate_or_dangling_ids_are_rejected(self):
        mutations = (
            lambda source: source["annotations"][1].update(id=10),
            lambda source: source["images"].append(copy.deepcopy(source["images"][0])),
            lambda source: source["annotations"][0].update(image_id=99),
        )
        for index, mutate in enumerate(mutations):
            source = source_coco("train")
            mutate(source)
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                derive_view(source, "train", strict=False)
        with self.assertRaises(ValueError):
            derive_view(source_coco("train"), "validation", strict=False)

    def test_prepare_is_idempotent_and_preserves_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_sources(root)
            before = {p: p.read_bytes() for p in (root / "annotations").iterdir()}
            first = prepare_views(root, strict=False)
            files = [root / block["view"] for block in first["splits"].values()]
            times = [p.stat().st_mtime_ns for p in files]
            self.assertEqual(prepare_views(root, strict=False), first)
            self.assertEqual(validate_views(root, strict=False), first)
            self.assertEqual([p.stat().st_mtime_ns for p in files], times)
            self.assertTrue(all(p.read_bytes() == data for p, data in before.items()))
            self.assertEqual(first["splits"]["train"]["original_category_counts"], {"1": 1, "2": 2})

    def test_interrupted_preparation_resumes_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_sources(root)
            calls = 0

            def interrupt(path, value):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("simulated interruption")
                pg.write_once_original(path, value)

            pg.write_once_original = pg.write_once
            try:
                with patch.object(pg, "write_once", side_effect=interrupt), self.assertRaises(RuntimeError):
                    prepare_views(root, strict=False)
                completed = prepare_views(root, strict=False)
                self.assertEqual(validate_views(root, strict=False), completed)
            finally:
                del pg.write_once_original

    def test_changed_source_or_view_is_not_overwritten(self):
        for target in ("source", "view"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.make_sources(root)
                manifest = prepare_views(root, strict=False)
                path = root / manifest["splits"]["train"][target]
                payload = json.loads(path.read_text())
                payload["annotations"][0]["bbox"][0] = 99
                path.write_text(json.dumps(payload))
                changed = path.read_bytes()
                with self.assertRaises((ValueError, FileExistsError)):
                    prepare_views(root, strict=False)
                with self.assertRaises((ValueError, FileExistsError)):
                    validate_views(root, strict=False)
                self.assertEqual(path.read_bytes(), changed)

    def test_smoke_contract_cannot_be_reused_after_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract = {"source": "same", "recipe": {"micro_batch_per_rank": 2, "accumulation_steps": 4, "effective_image_batch": 16, "train_workers_per_rank": 4, "val_workers_per_rank": 2, "ddp_accumulation_sync": "every_microbatch_plus_explicit_final_when_static_graph"}}
            path, _ = smoke_fixture(Path(tmp), contract)
            require_smoke(path, contract)
            with self.assertRaises(ValueError):
                require_smoke(path, {**contract, "source": "changed"})

    def test_forged_or_incomplete_smoke_evidence_is_rejected(self):
        mutations = (
            lambda report: report["ranks"][1].update(rank=0),
            lambda report: report["ranks"][0].update(device="Tesla V100"),
            lambda report: report["ranks"][0]["fixtures"].pop(),
            lambda report: report["ranks"][0]["fixtures"][0].update(mask_grad=0.0),
            lambda report: report["ranks"][0]["fixtures"][0].update(ddp_gradient_equal=False),
            lambda report: report["ranks"][0]["fixtures"][0].update(explicit_gradient_sync=False),
            lambda report: report["ranks"][0]["fixtures"][0]["output_shapes"].update(pred_masks=[4, 100, 288, 288]),
            lambda report: report["ranks"][0]["production_integration"].update(train_workers=0),
            lambda report: report["ranks"][1]["validation"].update(prediction_categories=[]),
            lambda report: report["ranks"][0]["validation"].update(checkpoint_fresh_model=False),
            lambda report: report["ranks"][0]["memory_mib"].update(peak_reserved=0.0),
            lambda report: report.update(skipped_checks=["workers"]),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index), tempfile.TemporaryDirectory() as tmp:
                contract = {"recipe": {"micro_batch_per_rank": 2, "accumulation_steps": 4, "effective_image_batch": 16, "train_workers_per_rank": 4, "val_workers_per_rank": 2, "ddp_accumulation_sync": "every_microbatch_plus_explicit_final_when_static_graph"}}
                path, report = smoke_fixture(Path(tmp), contract)
                mutate(report)
                path.write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    require_smoke(path, contract)

    def test_efficacy_declaration_requires_user_approval_and_frozen_a0(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a0 = root / "eval_tree.json"
            a0.write_text(json.dumps({"dataset_protocol": {"annotation_sha256": "a" * 64}}))
            declaration = {
                "version": 1, "arm": "prompt_granularity", "status": "approved",
                "approved_by": "user", "approved_utc": "2026-09-10T00:00:00+00:00",
                "single_seed_interpretation": "descriptive_unless_replicated",
                "a0_eval_sha256": file_digest(a0), "a0_annotation_sha256": "a" * 64,
                "criteria": [
                    {"role": "primary_effect", "metric": "Mask AP", "minimum_delta": 0.01},
                    {"role": "non_regression", "metric": "Bnd AP instance", "minimum_delta": -0.003},
                ],
            }
            path = root / "efficacy.json"
            path.write_text(json.dumps(declaration))
            self.assertEqual(validate_efficacy_spec(path, a0_eval_path=a0)["declaration"], declaration)
            for field, value in (("status", "draft"), ("approved_by", "assistant"), ("approved_utc", "2099-01-01T00:00:00+00:00"), ("a0_eval_sha256", "b" * 64)):
                with self.subTest(field=field):
                    invalid = copy.deepcopy(declaration)
                    invalid[field] = value
                    path.write_text(json.dumps(invalid))
                    with self.assertRaises(ValueError):
                        validate_efficacy_spec(path, a0_eval_path=a0)

    def test_resume_rejects_smoke_or_unrelated_checkpoint(self):
        manifest = {"purpose": "full", "contract": {"source": "same", "view": "same"}, "efficacy": {"sha": "same"}}
        state = {"model": {}, "optimizer": {}, "loss": {}, "scaler": {}, "epoch": 1, "steps": 2}
        verify_checkpoint_contract({**state, "prompt_granularity": manifest}, manifest)
        for checkpoint in (
            {},
            {**state, "prompt_granularity": {"purpose": "smoke", "contract": manifest["contract"]}},
            {**state, "prompt_granularity": {"purpose": "full", "contract": {"source": "other"}}},
            {"prompt_granularity": manifest},
        ):
            with self.assertRaises(ValueError):
                verify_checkpoint_contract(checkpoint, manifest)

    def test_smoke_checkpoint_uses_fresh_model_and_coordinated_equality(self):
        source = (CODEBASE / "Core/prompt_granularity_training.py").read_text()
        self.assertEqual(source.count("model(reload_inputs)"), 2)
        self.assertNotIn("ddp(reload_inputs)", source)
        self.assertIn("sync_context = _ddp_accumulation_context", source)
        self.assertIn("dist.broadcast_object_list(reload_payload, src=0, device=device)", source)
        self.assertIn("torch.distributed.all_gather(signatures, gradient_signature)", source)
        self.assertLess(source.index("explicit_gradient_sync = _synchronize_static_graph_accumulated_gradients"), source.index("scaler.unscale_"))
        trainer_source = (CODEBASE / "Core/progress_trainer.py").read_text()
        self.assertIn("ddp_context = _ddp_accumulation_context", trainer_source)
        self.assertLess(trainer_source.index("_synchronize_static_graph_accumulated_gradients(self.model"), trainer_source.index("self.scaler.unscale_"))
        self.assertNotIn("self.model.no_sync()", trainer_source)
        order = [
            source.index("dense_output ="),
            source.index("reload_inputs ="),
            source.index("before = _last_output"),
            source.index("checkpoint_path ="),
            source.index("del ddp, model, optim, criterion, scaler"),
            source.index("model = instantiate", source.index("del ddp, model, optim, criterion, scaler")),
            source.index("after = _last_output"),
            source.index("dist.all_reduce(equality"),
            source.index("rank_report ="),
        ]
        self.assertEqual(order, sorted(order))

    def test_smoke_report_has_a_read_only_cli_validator(self):
        result = subprocess.run(
            [sys.executable, str(CODEBASE / "Core/prompt_granularity.py"), "--help"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("verify-smoke", result.stdout)

    def test_smoke_cli_pins_worker_recipe_before_contract(self):
        device_policy = types.ModuleType("device_policy")
        device_policy.shared_node_env = lambda _: {}
        keys = (
            "PROMPT_GRANULARITY_ENABLED_OVERRIDE",
            "FULL_FT_RUN_NAME_OVERRIDE",
            "TRAIN_BATCH_SIZE_OVERRIDE",
            "FULL_FT_NUM_TRAIN_WORKERS_OVERRIDE",
            "FULL_FT_NUM_VAL_WORKERS_OVERRIDE",
            "ACT_CKPT_VISION_BACKBONE_OVERRIDE",
        )
        observed = {}

        def contract(_):
            observed.update({key: os.environ.get(key) for key in keys})
            return {"recipe": {}}

        argv = ["prompt_granularity.py", "verify-smoke", "--report", "smoke_report.json", "--checkpoint-backbone"]
        with patch.dict(os.environ, {}, clear=False), patch.dict(sys.modules, {"device_policy": device_policy}), patch.object(sys, "argv", argv), patch.object(pg, "training_contract", side_effect=contract), patch.object(pg, "require_smoke", return_value={}), patch("builtins.print"):
            pg.main()
        self.assertEqual(observed, {
            "PROMPT_GRANULARITY_ENABLED_OVERRIDE": "1",
            "FULL_FT_RUN_NAME_OVERRIDE": "sam3_prompt_granularity",
            "TRAIN_BATCH_SIZE_OVERRIDE": "2",
            "FULL_FT_NUM_TRAIN_WORKERS_OVERRIDE": "4",
            "FULL_FT_NUM_VAL_WORKERS_OVERRIDE": "2",
            "ACT_CKPT_VISION_BACKBONE_OVERRIDE": "1",
        })

    def test_validation_never_creates_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_sources(root)
            with self.assertRaises(FileNotFoundError):
                validate_views(root, strict=False)
            self.assertFalse((root / "reports").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
