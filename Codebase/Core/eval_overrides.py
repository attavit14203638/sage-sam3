"""Project overrides for SAM3 training and evaluation internals.

The classes replace upstream components that the locked recipe changes: the prediction post-processor
and dumper, the Boundary IoU evaluator, the semantic segmentation criterion, and the sampled mask
loss. Eight narrow patches, named in `PATCH_NAMES`, are then applied to the vendored SAM3 source at
import time so that the vendored source itself stays pristine: the cgF1 exhaustiveness flag and score
threshold, gradient-enabled `addmm_act`, an epsilon-safe generalised box IoU, a stable focal loss at
gamma 0, instance-chunked point sampling, single-rank-safe loss normalisation and barriers, and a
finite Hungarian cost matrix.

A patch that cannot be applied, for example because an optional component is unavailable, is logged
rather than fatal, and `patch_status()` reports the outcome of every patch. Import this module before
the trainer builds the model.
"""

from __future__ import annotations

import gc
import logging
from collections import defaultdict
from typing import Any

from sam3.eval.coco_writer import PredictionDumper
from sam3.eval.postprocessors import PostProcessImage
from sam3.train.utils.distributed import all_gather, gather_to_rank_0_via_filesys
from sam3.train.loss.loss_fns import Masks, SemanticSegCriterion



class ProjectPostProcessImage(PostProcessImage):
    """Post-processor that returns original-size masks as RLE when scoring segmentation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("iou_type", "bbox") == "segm":
            kwargs["use_original_sizes_mask"] = True
            kwargs["convert_mask_to_rle"] = True
        super().__init__(*args, **kwargs)


class ProjectPredictionDumper(PredictionDumper):
    """Prediction dumper that merges per-rank predictions and rejects masks exported at a 1 by 1 size."""

    def gather_and_merge_predictions(self):
        """Gather predictions from every rank and keep the highest-scoring `maxdets` per image."""
        logging.info("Prediction Dumper: Gathering predictions from all processes")
        gc.collect()

        if self.gather_pred_via_filesys:
            gathered = gather_to_rank_0_via_filesys(self.dump)
        else:
            gathered = all_gather(self.dump, force_cpu=True)

        preds_by_image = defaultdict(list)
        for rank_dump in gathered:
            for pred in rank_dump:
                preds_by_image[pred["image_id"]].append(pred)

        merged_dump = []
        for image_id in sorted(preds_by_image):
            preds = preds_by_image[image_id]
            preds.sort(key=lambda pred: pred["score"], reverse=True)
            if self.maxdets > 0:
                preds = preds[: self.maxdets]
            merged_dump.extend(preds)

        return merged_dump

    def prepare_for_coco_segmentation(self, predictions):
        """Build COCO segmentation results; raise if a mask was exported at 1 by 1 while its box is in image coordinates."""
        results = super().prepare_for_coco_segmentation(predictions)
        bad_results = [
            result
            for result in results
            if tuple(result.get("segmentation", {}).get("size", ())) == (1, 1)
            and result.get("bbox")
            and max(result["bbox"]) > 1
        ]
        if bad_results:
            raise ValueError(
                "Segmentation predictions were exported as 1x1 masks while boxes are in image coordinates. "
                "This indicates validation mask postprocessing is using normalized mask sizes instead of original image sizes."
            )
        return results


class BoundaryIoUEvaluator:
    """Mean per-image Boundary IoU between the union of all ground-truth masks and the union of all predictions.

    This semantic-union diagnostic is distinct from instance Boundary AP.
    """

    def __init__(self, gt_path: str, iou_type: str = "segm", dilation_ratio: float = 0.02) -> None:
        self.gt_path = gt_path
        self.iou_type = iou_type
        self.dilation_ratio = dilation_ratio

    def evaluate(self, dumped_file: str) -> dict[str, float]:
        """Load the ground truth and a prediction dump, and return `coco_eval_<iou_type>_boundary_iou` averaged over images."""
        import json
        import numpy as np
        from pycocotools.coco import COCO
        import pycocotools.mask as mask_utils
        from scipy.ndimage import binary_erosion

        logging.info("BoundaryIoU evaluator: Loading groundtruth")
        coco_gt = COCO(self.gt_path)

        logging.info("BoundaryIoU evaluator: Loading predictions")
        with open(dumped_file, "r") as f:
            predictions = json.load(f)

        # Group predictions by image_id
        img_to_preds = defaultdict(list)
        for pred in predictions:
            img_to_preds[pred["image_id"]].append(pred)

        img_ids = sorted(coco_gt.getImgIds())

        total_biou = 0.0
        count = 0

        for img_id in img_ids:
            img_info = coco_gt.loadImgs(img_id)[0]
            h, w = img_info["height"], img_info["width"]

            # Reconstruct GT semantic mask (union of all instance masks)
            ann_ids = coco_gt.getAnnIds(imgIds=img_id)
            anns = coco_gt.loadAnns(ann_ids)
            if anns:
                gt_rles = [coco_gt.annToRLE(ann) for ann in anns]
                gt_mask = mask_utils.decode(mask_utils.merge(gt_rles, intersect=False)).astype(bool)
            else:
                gt_mask = np.zeros((h, w), dtype=bool)

            # Reconstruct Pred semantic mask (union of all predicted instance masks)
            preds = img_to_preds.get(img_id, [])
            if preds:
                pred_rles = [pred["segmentation"] for pred in preds]
                pred_mask = mask_utils.decode(mask_utils.merge(pred_rles, intersect=False)).astype(bool)
            else:
                pred_mask = np.zeros((h, w), dtype=bool)

            # Boundary IoU per Cheng et al. 2021 ("Boundary IoU: Improving
            # Object-Centric Image Segmentation Evaluation"):
            #
            #   B-IoU = |(G_d ∩ G) ∩ (P_d ∩ P)| / |(G_d ∩ G) ∪ (P_d ∩ P)|
            #
            # where G_d ∩ G is the *inner* band of mask G (pixels of G within
            # distance d of the contour), computed on a discrete grid as
            # `M AND NOT erosion(M, d)`.
            #
            # The band geometry below is byte-aligned with the reference
            # implementation, `bowenc0221/boundary-iou-api`
            # (`boundary_iou/utils/boundary_utils.py::mask_to_boundary`), so
            # this diagnostic and the reported Boundary AP measure the same
            # band. Three details are load-bearing (see the evaluation protocol):
            #
            #   1. d = max(1, round(ratio * diagonal)). The reference rounds, which
            #      gives 58 at ratio 0.02 (truncation would give 57); both give 8 at
            #      the 8px fine band.
            #   2. A full 3x3 structuring element, i.e. Chebyshev (L-inf)
            #      distance. `generate_binary_structure(2, 1)` is 4-connected
            #      and yields L1 distance, a different band shape.
            #   3. `border_value=0`, so the image border counts as background
            #      and a mask truncated by it carries a band there. The
            #      reference achieves this with an explicit zero pad
            #      (`copyMakeBorder(..., value=0)`) because cv2.erode does not
            #      erode from the array edge by default. This also makes the
            #      all-foreground case correct without a special branch: an
            #      all-True mask yields the d-wide frame, not an empty band.
            pred_any = pred_mask.any()
            gt_any = gt_mask.any()
            if not gt_any and not pred_any:
                biou = 1.0
            else:
                dilation_pixels = max(1, int(round(float(np.hypot(h, w)) * self.dilation_ratio)))
                struct = np.ones((3, 3), dtype=bool)

                if gt_any:
                    gt_eroded = binary_erosion(
                        gt_mask, structure=struct, iterations=dilation_pixels, border_value=0
                    )
                    gt_boundary = gt_mask & ~gt_eroded
                else:
                    gt_boundary = np.zeros_like(gt_mask, dtype=bool)

                if pred_any:
                    pred_eroded = binary_erosion(
                        pred_mask, structure=struct, iterations=dilation_pixels, border_value=0
                    )
                    pred_boundary = pred_mask & ~pred_eroded
                else:
                    pred_boundary = np.zeros_like(pred_mask, dtype=bool)

                intersection = np.sum(gt_boundary & pred_boundary)
                union = np.sum(gt_boundary | pred_boundary)

                if union == 0:
                    biou = 1.0 if np.all(gt_mask == pred_mask) else 0.0
                else:
                    biou = float(intersection / union)

            total_biou += biou
            count += 1

        mean_biou = total_biou / count if count > 0 else 0.0
        metric_name = f"coco_eval_{self.iou_type}_boundary_iou"
        logging.info("BoundaryIoU evaluator: %s = %.6f", metric_name, mean_biou)
        return {metric_name: mean_biou}


class ProjectSemanticSegCriterion(SemanticSegCriterion):
    """Semantic segmentation criterion that scores a batch without annotations against an all-background target."""

    def get_loss(self, out_dict, targets):
        """Return the semantic segmentation loss, supplying an empty target when the batch has no masks or boxes."""
        from sam3.train.loss.loss_fns import SemanticSegCriterion
        import torch

        outputs = out_dict["semantic_seg"]
        if (
            "semantic_masks" not in targets
            or targets["semantic_masks"] is None
            or targets["semantic_masks"].size(0) == 0
        ):
            if targets["num_boxes"].sum() == 0:
                # Handle empty/zero-annotation (true negative) image batches on distributed filesystems
                device = outputs.device
                B = outputs.shape[0]
                if self.downsample:
                    h, w = outputs.shape[-2:]
                else:
                    h, w = 1008, 1008
                semantic_targets = torch.zeros((B, h, w), dtype=torch.bool, device=device)

                mock_targets = targets.copy()
                mock_targets["semantic_masks"] = semantic_targets
                return super().get_loss(out_dict, mock_targets)

        return super().get_loss(out_dict, targets)


class ProjectSampledMasks(Masks):
    """Upstream sampled mask loss with shape and allocation safety checks.

    The wrapper normalises mask tensor rank, records peak instance count against the
    `grid_sample` int32 element limit, and returns differentiable zero losses when both the
    prediction and target sets are empty. Native mask and Dice objectives remain unchanged.
    """

    @staticmethod
    def _to_3d_mask_tensor(masks, name: str, take_first_channel: bool = False):
        original_shape = tuple(masks.shape)
        if masks.ndim == 2:
            return masks.unsqueeze(0)
        if masks.ndim >= 4:
            for dim in range(masks.ndim - 1, 0, -1):
                if masks.shape[dim] == 1:
                    masks = masks.squeeze(dim)
        if masks.ndim == 3:
            return masks
        if take_first_channel and masks.ndim == 4 and masks.shape[1] > 0:
            masks = masks[:, 0]
            if masks.ndim == 3:
                return masks
        raise AssertionError(
            f"{name} must reduce to [N,H,W] for sampled mask loss, got {original_shape} -> {tuple(masks.shape)}"
        )

    # Log each new high-water mark so allocations approaching the `grid_sample` int32 element
    # limit are visible before the limit is crossed.
    _peak_instances = 0

    def _record_instance_count(self, target_masks) -> None:
        n = int(target_masks.shape[0])
        if n <= type(self)._peak_instances:
            return
        type(self)._peak_instances = n
        elements = n * int(target_masks.shape[-2]) * int(target_masks.shape[-1])
        logging.info(
            "mask loss: new peak instance count N=%d (%.2f G elements at %dx%d, "
            "%.0f%% of the grid_sample int32 limit)",
            n, elements / 1e9, int(target_masks.shape[-2]), int(target_masks.shape[-1]),
            100.0 * elements / (2**31 - 1),
        )

    def _sampled_loss(self, src_masks, target_masks, num_boxes):
        if src_masks.shape[0] == 0 or target_masks.shape[0] == 0:
            if src_masks.shape[0] == 0 and target_masks.shape[0] == 0:
                zero = src_masks.float().sum() * 0.0
                return {"loss_mask": zero, "loss_dice": zero}
            raise AssertionError(
                f"sampled mask loss received mismatched empty masks: src_masks={tuple(src_masks.shape)}, target_masks={tuple(target_masks.shape)}"
            )
        src_masks = self._to_3d_mask_tensor(src_masks, "src_masks", take_first_channel=True)
        target_masks = self._to_3d_mask_tensor(target_masks, "target_masks")
        self._record_instance_count(target_masks)
        return super()._sampled_loss(src_masks, target_masks, num_boxes)


# =============================================================================
# Monkey-patches to keep external/vendored libraries (Support/sam3) pristine
# =============================================================================

PATCH_NAMES = (
    "cgf1_exhaustive_flag",
    "addmm_act_grad",
    "box_giou_epsilon",
    "focal_loss_gamma_zero",
    "point_sample_chunking",
    "single_rank_collectives",
    "matcher_finite_cost",
    "cgf1_score_threshold",
)
_PATCH_STATUS: dict[str, str] = {}


def _record_patch(name: str, error: Exception | None = None) -> None:
    """Record whether the patch `name` was applied; a failure never stops the import."""
    _PATCH_STATUS[name] = "applied" if error is None else f"failed: {type(error).__name__}: {error}"


def patch_status() -> dict[str, str]:
    """Return `applied`, `failed: <reason>`, or `not attempted` for each patch, in application order."""
    return {name: _PATCH_STATUS.get(name, "not attempted") for name in PATCH_NAMES}


# 1. Patch COCOCustom in sam3.eval.cgf1_eval to support standard COCO jsons
# that do not contain the custom 'is_instance_exhaustive' key.
try:
    from sam3.eval.cgf1_eval import COCOCustom
    original_cococustom_init = COCOCustom.__init__

    def patched_cococustom_init(self, annotation_file=None):
        original_cococustom_init(self, annotation_file)
        if "images" in self.dataset:
            for img in self.dataset["images"]:
                if "is_instance_exhaustive" not in img:
                    img["is_instance_exhaustive"] = True

    COCOCustom.__init__ = patched_cococustom_init
    _record_patch("cgf1_exhaustive_flag")
except Exception as e:
    _record_patch("cgf1_exhaustive_flag", e)
    logging.warning(f"Failed to monkey-patch COCOCustom: {e}")


# 2. Patch addmm_act in sam3.perflib.fused to allow training with gradients enabled.
try:
    import sys
    import sam3.perflib.fused
    import torch
    original_addmm_act = sam3.perflib.fused.addmm_act

    def patched_addmm_act(activation, linear, mat1):
        if torch.is_grad_enabled():
            x = linear(mat1)
            if activation in [torch.nn.functional.relu, torch.nn.ReLU]:
                return torch.nn.functional.relu(x)
            if activation in [torch.nn.functional.gelu, torch.nn.GELU]:
                return torch.nn.functional.gelu(x)
            raise ValueError(f"Unexpected activation {activation}")
        return original_addmm_act(activation, linear, mat1)

    sam3.perflib.fused.addmm_act = patched_addmm_act
    if "sam3.model.vitdet" in sys.modules:
        sys.modules["sam3.model.vitdet"].addmm_act = patched_addmm_act
    _record_patch("addmm_act_grad")
except Exception as e:
    _record_patch("addmm_act_grad", e)
    logging.warning(f"Failed to monkey-patch addmm_act: {e}")


try:
    import torch as _torch
    import sam3.model.box_ops as _box_ops
    import sam3.train.matcher as _matcher_mod_for_box_ops

    def _safe_box_area_xyxy(boxes):
        x0, y0, x1, y1 = boxes.unbind(-1)
        return (x1 - x0).clamp(min=0) * (y1 - y0).clamp(min=0)

    def _safe_generalized_box_iou(boxes1, boxes2, eps=1e-7):
        area1 = _safe_box_area_xyxy(boxes1)
        area2 = _safe_box_area_xyxy(boxes2)
        lt = _torch.max(boxes1[..., :, None, :2], boxes2[..., None, :, :2])
        rb = _torch.min(boxes1[..., :, None, 2:], boxes2[..., None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        union = (area1[..., None] + area2[..., None, :] - inter).clamp_min(eps)
        iou = inter / union
        lt = _torch.min(boxes1[..., :, None, :2], boxes2[..., None, :, :2])
        rb = _torch.max(boxes1[..., :, None, 2:], boxes2[..., None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        enclosing_area = (wh[..., 0] * wh[..., 1]).clamp_min(eps)
        giou = iou - (enclosing_area - union) / enclosing_area
        return _torch.nan_to_num(giou, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1, 1)

    def _safe_fast_diag_generalized_box_iou(boxes1, boxes2, eps=1e-7):
        if boxes1.numel() == 0:
            return boxes1.new_zeros((0,))
        box1_xy = boxes1[:, 2:]
        box1_XY = boxes1[:, :2]
        box2_xy = boxes2[:, 2:]
        box2_XY = boxes2[:, :2]
        area1 = (box1_xy - box1_XY).clamp(min=0).prod(-1)
        area2 = (box2_xy - box2_XY).clamp(min=0).prod(-1)
        lt = _torch.max(box1_XY, box2_XY)
        rb = _torch.min(box1_xy, box2_xy)
        inter = (rb - lt).clamp(min=0).prod(-1)
        union = (area1 + area2 - inter).clamp_min(eps)
        iou = inter / union
        lt2 = _torch.min(box1_XY, box2_XY)
        rb2 = _torch.max(box1_xy, box2_xy)
        enclosing_area = (rb2 - lt2).clamp(min=0).prod(-1).clamp_min(eps)
        giou = iou - (enclosing_area - union) / enclosing_area
        return _torch.nan_to_num(giou, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1, 1)

    _box_ops.generalized_box_iou = _safe_generalized_box_iou
    _box_ops.fast_diag_generalized_box_iou = _safe_fast_diag_generalized_box_iou
    _matcher_mod_for_box_ops.generalized_box_iou = _safe_generalized_box_iou
    logging.info("Patched box GIoU helpers with epsilon-safe project overrides.")
    _record_patch("box_giou_epsilon")
except Exception as e:
    _record_patch("box_giou_epsilon", e)
    logging.warning(f"Failed to monkey-patch box GIoU helpers: {e}")


try:
    import torch.nn.functional as _F
    import sam3.train.loss.loss_fns as _loss_fns

    _orig_sigmoid_focal_loss = _loss_fns.sigmoid_focal_loss

    def _stable_sigmoid_focal_loss(
        inputs,
        targets,
        num_boxes,
        alpha: float = 0.25,
        gamma: float = 2,
        loss_on_multimask=False,
        reduce=True,
        triton=True,
    ):
        if gamma != 0:
            return _orig_sigmoid_focal_loss(
                inputs,
                targets,
                num_boxes,
                alpha=alpha,
                gamma=gamma,
                loss_on_multimask=loss_on_multimask,
                reduce=reduce,
                triton=triton,
            )

        loss = _F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
        if not reduce:
            return loss
        if loss_on_multimask:
            return loss.flatten(2).mean(-1) / num_boxes
        return loss.mean(1).sum() / num_boxes

    _loss_fns.sigmoid_focal_loss = _stable_sigmoid_focal_loss
    logging.info("Patched sigmoid_focal_loss gamma=0 path with stable BCE project override.")
    _record_patch("focal_loss_gamma_zero")
except Exception as e:
    _record_patch("focal_loss_gamma_zero", e)
    logging.warning(f"Failed to monkey-patch sigmoid_focal_loss gamma=0 path: {e}")


# `grid_sample` uses 32-bit indexing, so a dense [N, 1, H, W] mask stack can exceed its
# addressable element count before GPU memory is exhausted. Chunking over the independent
# instance dimension preserves numerical results and consumes no random state because point
# coordinates are generated before this call. `loss_fns` binds `point_sample` at import time,
# so both the defining module and the imported binding must be patched.
_POINT_SAMPLE_SAFE_ELEMENTS = (2**31 - 1) // 2  # 2x margin under the 32-bit limit
_POINT_SAMPLE_MAX_CHUNK = 256  # also caps grid_sample's fp32 promotion of a big fp16 stack
_point_sample_chunk_warned = False

try:
    import torch as _torch_ps
    import sam3.train.loss.loss_fns as _loss_fns_ps
    import sam3.train.loss.mask_sampling as _mask_sampling_ps

    _original_point_sample = _mask_sampling_ps.point_sample

    def _chunked_point_sample(input, point_coords, **kwargs):
        """`point_sample`, chunked over instances when the input would overflow int32."""
        global _point_sample_chunk_warned
        n = int(input.shape[0])
        if n <= 1 or input.numel() <= _POINT_SAMPLE_SAFE_ELEMENTS:
            return _original_point_sample(input, point_coords, **kwargs)

        per_instance = max(1, input.numel() // n)
        chunk = max(1, min(_POINT_SAMPLE_MAX_CHUNK, _POINT_SAMPLE_SAFE_ELEMENTS // per_instance))
        if not _point_sample_chunk_warned:
            _point_sample_chunk_warned = True
            logging.warning(
                "point_sample: chunking a %s input (%.2f G elements, N=%d) at %d instances "
                "per chunk -- unchunked this exceeds grid_sample's 32-bit index limit. "
                "The result is bit-identical; see eval_overrides.py.",
                tuple(input.shape), input.numel() / 1e9, n, chunk,
            )
        outs = [
            _original_point_sample(input[i : i + chunk], point_coords[i : i + chunk], **kwargs)
            for i in range(0, n, chunk)
        ]
        return _torch_ps.cat(outs, dim=0)

    _mask_sampling_ps.point_sample = _chunked_point_sample
    _loss_fns_ps.point_sample = _chunked_point_sample
    logging.info(
        "Patched point_sample to chunk inputs above %.2f G elements (grid_sample int32 limit).",
        _POINT_SAMPLE_SAFE_ELEMENTS / 1e9,
    )
    _record_patch("point_sample_chunking")
except Exception as e:
    _record_patch("point_sample_chunking", e)
    logging.warning(f"Failed to monkey-patch point_sample instance chunking: {e}")


try:
    import torch as _torch_dist_safe
    import sam3.train.loss.sam3_loss as _sam3_loss_mod
    import sam3.train.trainer as _trainer_mod
    import sam3.train.utils.distributed as _distributed_mod

    _orig_distributed_barrier = _distributed_mod.barrier
    _orig_torch_distributed_barrier = _torch_dist_safe.distributed.barrier

    def _single_rank_safe_get_num_boxes(self, targets):
        if self.normalize_by_valid_object_num:
            boxes_hw = targets["boxes"].view(-1, 4)
            num_boxes = (boxes_hw[:, 2:] > 0).all(dim=-1).sum().float()
        else:
            num_boxes = targets["num_boxes"].sum().float()
        if self.normalization == "global":
            world_size = 1
            if _torch_dist_safe.distributed.is_available() and _torch_dist_safe.distributed.is_initialized():
                world_size = _torch_dist_safe.distributed.get_world_size()
            if world_size > 1:
                _torch_dist_safe.distributed.all_reduce(num_boxes)
            num_boxes = _torch_dist_safe.clamp(num_boxes / world_size, min=1)
        elif self.normalization == "local":
            num_boxes = _torch_dist_safe.clamp(num_boxes, min=1)
        elif self.normalization == "none":
            num_boxes = 1
        return num_boxes

    def _single_rank_safe_barrier():
        if (
            _torch_dist_safe.distributed.is_available()
            and _torch_dist_safe.distributed.is_initialized()
            and _torch_dist_safe.distributed.get_world_size() <= 1
        ):
            return
        return _orig_distributed_barrier()

    def _single_rank_safe_torch_barrier(*args, **kwargs):
        if (
            _torch_dist_safe.distributed.is_available()
            and _torch_dist_safe.distributed.is_initialized()
            and _torch_dist_safe.distributed.get_world_size() <= 1
        ):
            return
        return _orig_torch_distributed_barrier(*args, **kwargs)

    _sam3_loss_mod.Sam3LossWrapper._get_num_boxes = _single_rank_safe_get_num_boxes
    _distributed_mod.barrier = _single_rank_safe_barrier
    _trainer_mod.barrier = _single_rank_safe_barrier
    _trainer_mod.dist.barrier = _single_rank_safe_torch_barrier
    logging.info("Patched single-rank loss normalization and barriers to avoid no-op NCCL collectives.")
    _record_patch("single_rank_collectives")
except Exception as e:
    _record_patch("single_rank_collectives", e)
    logging.warning(f"Failed to monkey-patch single-rank distributed no-ops: {e}")


# 3. Patch _do_matching in sam3.train.matcher to guard the Hungarian cost
# matrix against non-finite (NaN/Inf) entries.
#
# scipy.optimize.linear_sum_assignment raises
#   "ValueError: matrix contains invalid numeric entries"
# whenever the cost matrix contains a NaN or Inf. A single degenerate sample
# can poison one batch's cost matrix late in training via two known sources:
#   (a) generalized_box_iou (sam3/model/box_ops.py): a predicted or GT box
#       with zero width/height makes union/area == 0 -> 0/0 == NaN; and
#   (b) the focal cost_class term (matcher.py): torch.log(out_prob) /
#       torch.log(1 - out_prob) -> -inf when a query probability saturates to
#       exactly 0 or 1.
# We replace every non-finite entry with a large finite cost (1e9) so the
# offending pairing is simply never selected by the assignment solver,
# mirroring the library's own 1e9 "invalid output/target" convention. This
# keeps Support/sam3 pristine and lets training proceed instead of aborting.
try:
    import numpy as _np
    import sam3.train.matcher as _matcher_mod

    _orig_do_matching = _matcher_mod._do_matching

    def _safe_do_matching(cost, *args, **kwargs):
        if cost is not None and not _np.all(_np.isfinite(cost)):
            n_bad = int((~_np.isfinite(_np.asarray(cost))).sum())
            logging.warning(
                "Matcher cost matrix contained %d non-finite entries; "
                "sanitizing to 1e9 before linear_sum_assignment.",
                n_bad,
            )
            cost = _np.nan_to_num(cost, nan=1e9, posinf=1e9, neginf=1e9)
        return _orig_do_matching(cost, *args, **kwargs)

    _matcher_mod._do_matching = _safe_do_matching

    import torch as _torch_matcher_diag

    def _log_nonfinite_matcher_tensor(name, tensor):
        if tensor is None or not _torch_matcher_diag.is_tensor(tensor):
            return
        try:
            detached = tensor.detach()
            finite = _torch_matcher_diag.isfinite(detached)
            if bool(finite.all().item()):
                return
            count = getattr(_log_nonfinite_matcher_tensor, "_count", 0)
            if count >= 80:
                return
            setattr(_log_nonfinite_matcher_tensor, "_count", count + 1)
            finite_values = detached[finite]
            finite_min = float(finite_values.min().cpu().item()) if finite_values.numel() else None
            finite_max = float(finite_values.max().cpu().item()) if finite_values.numel() else None
            logging.warning(
                "Non-finite matcher tensor %s: shape=%s nonfinite=%d/%d finite_min=%s finite_max=%s.",
                name,
                tuple(detached.shape),
                int((~finite).sum().cpu().item()),
                int(detached.numel()),
                finite_min,
                finite_max,
            )
        except Exception:
            logging.warning("Failed to summarize matcher tensor %s", name, exc_info=True)

    def _log_nonfinite_matcher_inputs(outputs, batched_targets):
        for key in ("pred_logits", "pred_boxes", "pred_boxes_xyxy", "presence_logit_dec"):
            _log_nonfinite_matcher_tensor(f"outputs[{key}]", outputs.get(key))
        for key in ("boxes", "boxes_padded", "boxes_xyxy", "num_boxes"):
            _log_nonfinite_matcher_tensor(f"targets[{key}]", batched_targets.get(key))

    _orig_binary_hungarian_v2_forward = _matcher_mod.BinaryHungarianMatcherV2.forward

    def _patched_binary_hungarian_v2_forward(self, outputs, batched_targets, *args, **kwargs):
        _log_nonfinite_matcher_inputs(outputs, batched_targets)
        return _orig_binary_hungarian_v2_forward(self, outputs, batched_targets, *args, **kwargs)

    _matcher_mod.BinaryHungarianMatcherV2.forward = _patched_binary_hungarian_v2_forward

    _orig_binary_o2m_forward = _matcher_mod.BinaryOneToManyMatcher.forward

    def _patched_binary_o2m_forward(self, outputs, batched_targets, *args, **kwargs):
        _log_nonfinite_matcher_inputs(outputs, batched_targets)
        return _orig_binary_o2m_forward(self, outputs, batched_targets, *args, **kwargs)

    _matcher_mod.BinaryOneToManyMatcher.forward = _patched_binary_o2m_forward
    _record_patch("matcher_finite_cost")
except Exception as e:
    _record_patch("matcher_finite_cost", e)
    logging.warning(f"Failed to monkey-patch _do_matching: {e}")


# 4. Use the project operating-point score threshold for the in-training cgF1
# meter. This affects metric computation only; prediction export remains
# controlled separately by EVAL_DETECTION_THRESHOLD.
try:
    from config import EVAL_CGF1_SCORE_THRESHOLD as _CGF1_THRESH
    from sam3.eval.cgf1_eval import CGF1Eval as _CGF1Eval

    _orig_cgf1_init = _CGF1Eval.__init__

    def _patched_cgf1_init(self, *args, **kwargs):
        kwargs.pop("threshold", None)
        _orig_cgf1_init(self, *args, **kwargs)
        self.threshold = _CGF1_THRESH

    _CGF1Eval.__init__ = _patched_cgf1_init
    logging.info(
        "Patched CGF1Eval score threshold to %.3f for in-training cgF1 meter.",
        _CGF1_THRESH,
    )
    _record_patch("cgf1_score_threshold")
except Exception as e:
    _record_patch("cgf1_score_threshold", e)
    logging.warning(f"Failed to monkey-patch CGF1Eval threshold: {e}")

logging.info(
    "Project patches applied: %d of %d.",
    sum(status == "applied" for status in patch_status().values()),
    len(PATCH_NAMES),
)
