# Codebase

Runnable project code. See the repository root `README.md` for installation, `RESULTS.md` for
measured results, and `PROTOCOL.md` for the locked evaluation rules.

## Layout

```text
Core/          project-owned training, inference, evaluation, and protocol modules
Tests/         standalone regression suites; Tests/run_all.py runs them all
Notebook/      asset preparation and the inert experiment pipeline
Support/       external SAM3 source, supplied separately
runtime/       generated data, checkpoints, and inference outputs
experiments/   generated run configurations, logs, and model exports
```

`Support/sam3` is the SAM3 Python source, not the pretrained weights. Pretrained weights and
OAM-TCD data come from their original providers and are cached by the asset notebook.

## Core modules

Each module opens with a docstring that states its role, and public functions are documented in
place.

| Module | Responsibility |
|---|---|
| `config.py` | Central training and evaluation configuration |
| `project_paths.py` | Project path definitions and overwrite guards |
| `atomic_io.py` | Atomic writes for run artefacts |
| `project_workflows.py` | Asset, training, inference, evaluation, and launch workflows |
| `dataset_adapter.py` | OAM-TCD validation and COCO export |
| `project_torch_dataset.py` | Dataset wrapper used by the SAM3 trainer |
| `augmentation.py` | Training augmentation and verification |
| `training_pipeline.py` | Training entry point and generated SAM3 configuration |
| `progress_trainer.py` | Progress reporting, finite-value guards, and checkpoint views |
| `prompt_granularity.py` | Category views, recipe contracts, internal efficacy-screen checks, and run manifests |
| `prompt_granularity_training.py` | Checked prompt collator, loss normalisation, and distributed smoke test |
| `inference_engine.py` | Backend-neutral inference interface |
| `tiled_inference.py` | Sliding-window inference and cross-tile mask non-maximum suppression |
| `eval_predictions.py` | COCO, Boundary AP, cgF1, and diagnostic evaluation |
| `eval_overrides.py` | Shape, numerical, and allocation safety patches for SAM3, with `patch_status()` |
| `boundary_band.py` | Instance-relative Boundary AP parameterisation |
| `oam_tcd_protocol.py` | Dataset and annotation identity validation |
| `visualization.py` | Ground-truth and prediction rendering |
| `train_maskrcnn.py` | Mask R-CNN reference training |
| `eval_official_maskrcnn.py` | Mask R-CNN reference evaluation |
| `device_policy.py` | Precision, hardware, capacity, and comparability checks |

`A0` appears in a few constant, path, and artefact names (for example `A0_RUN_NAME`). It denotes
the Unified-Prompt SAM3 baseline.

## Tests

```bash
python -B Tests/run_all.py
```

Suites requiring unavailable dependencies are reported as skipped rather than passed. On the
complete GPU environment, use:

```bash
python -B Tests/run_all.py --strict-deps
```

`test_public_surface.py` checks, without any GPU stack, that every Core name imported by Core, the
tests, or the notebooks still exists. `test_eval_overrides.py` requires all eight SAM3 patches to
have applied whenever CUDA is available.

## Runs

All constants live in `Core/config.py`. Registering an arm does not activate it. A
prompt-granularity launch additionally requires a passing smoke report, the approved internal
efficacy-screen record, exact recipe agreement, supported hardware, and an explicit launch decision. Every
run writes a generated configuration and provenance record under its own experiment directory.

The committed notebooks are inert. They do not launch training, overwrite predictions, or
recompute evaluations unless their explicit controls are changed after the required checks.
