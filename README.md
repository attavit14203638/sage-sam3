# SAGE-SAM3

A Scale-Aligned, Granularity-Explicit Framework for Evaluating and Adapting a Vision Foundation Model for Tree Crown Instance Segmentation.

A complete, provenance-tracked system for fine-tuning and evaluating
[SAM3](https://ai.meta.com/sam3) for individual tree-crown instance segmentation on the
[OAM-TCD](https://huggingface.co/datasets/restor/tcd) aerial tree-cover dataset
(2048 by 2048 RGB tiles at 10 cm per pixel).

The repository contains two trained configurations and the locked evaluation protocol used
to compare them:

| Configuration | Description |
|---|---|
| **Unified-Prompt SAM3** (`sam3_crop_frts`, internal A0) | Full SAM3 fine-tune on all 4,169 OAM-TCD training images under the locked recipe. Separate masks share one `tree` training prompt. |
| **Granularity-aware SAM3** (`sam3_prompt_granularity`) | Same recipe, architecture, and inference. Training preserves the dataset's two annotation granularities (`tree` = individual crowns, `tree canopy` = unresolved canopy groups) as two text-conditioned queries instead of one unified concept. |

Final measured results are in [RESULTS.md](RESULTS.md); the evaluation rules are in
[PROTOCOL.md](PROTOCOL.md).

## What is included

```text
Codebase/
  Core/        training, tiled inference, evaluation, and protocol modules
  Notebook/    asset preparation and inert experiment pipeline
  Tests/       standalone regression suites (Tests/run_all.py runs them all)
  Support/     not included; external SAM3 source goes here (see Install)
  runtime/     generated at runtime: data exports, checkpoints, inference outputs
  experiments/ generated at runtime: per-run configs, logs, checkpoints
Context/
  active/      pre-registered efficacy declaration used by the experiment notebook
LICENSE        MIT licence for the original code
NOTICE         third-party notices
third_party/   licence text that accompanies adapted SAM3 portions
```

## Install

Python 3.11, CUDA 12.x.

```bash
git clone <this-repository> oam_tcd_sam3 && cd oam_tcd_sam3/Codebase
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip wheel
```

The SAM3 Python package is external and not redistributed here. Obtain the SAM3 source and
place it at `Codebase/Support/sam3/`, then:

```bash
pip install -r requirements.txt        # includes: pip install -e ./Support/sam3[train]
pip install git+https://github.com/facebookresearch/detectron2.git@v0.6   # Mask R-CNN reference only
huggingface-cli login                  # for the OAM-TCD dataset and SAM3 checkpoint
```

## Usage

1. **Prepare assets.** `Notebook/assets_preparation.ipynb` downloads/caches the SAM3
   checkpoint (`facebook/sam3`) and the OAM-TCD dataset, and creates the COCO-format export
   under `runtime/data/oam_tcd_instance_coco/`.

2. **Train.** Use `Notebook/experiment_pipeline.ipynb`. The `ARM` switch selects the
   configuration: `ARM = None` reproduces Unified-Prompt SAM3 (internal A0); `ARM = "prompt_granularity"` trains the
   granularity-aware configuration. Training is notebook-driven and guarded: launches
   require fresh headroom, contract, and (for the granularity arm) smoke and efficacy checks.

3. **Infer and evaluate.** The same notebook runs locked tiled inference (1024/256 tiling,
   mask-NMS) and the evaluation in PROTOCOL.md, then prints the comparison table against the
   Unified-Prompt SAM3 reference.

All training constants live in `Core/config.py`; every launch writes a per-run generated
config under its experiment folder.

## Tests

```bash
cd Codebase
python -B Tests/run_all.py               # everything present in this environment
python -B Tests/run_all.py --strict-deps # fail on dependency-skipped suites (GPU host)
```

Suites that need CUDA (device policy, eval overrides, prompt-granularity training,
progress trainer, resume staging) are skipped on machines without PyTorch rather than
counted as passes.

## Notes

- Results use one training seed (42) and are descriptive. No standard deviation or seed-variability claim is reported. TEST is used for in-training monitoring and final reporting, not as an untouched model-selection holdout.
- The reported model is always the last-epoch export, never the optional best checkpoint.
- Final inference uses only `tree`; canopy-prompt performance was not evaluated or reported.
- `runtime/` and `experiments/` outputs are intentionally not part of the repository.

## License and citation

The original code in this repository is released under the [MIT License](LICENSE). SAM3,
Detectron2, the Boundary IoU reference implementation, and the OAM-TCD data keep their own
licences and are not redistributed here; see [NOTICE](NOTICE). Portions of `Codebase/Core` that
adapt SAM3 code remain subject to the SAM License ([third_party/SAM_LICENSE.txt](third_party/SAM_LICENSE.txt)).
Research that uses SAM3 must acknowledge that use when it is published.

A citation entry will be added when the article is published.
