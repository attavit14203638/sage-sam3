# Results

All numbers are produced by the locked evaluation protocol in [PROTOCOL.md](PROTOCOL.md) on
the OAM-TCD TEST split: 439 images, 25,695 individual tree crowns scored, 4,951 canopy
groups ignored via COCO crowd semantics. Models report last-epoch weights.

## Individual-tree Mask AP

| | Mask AP | AP50 | AP75 | AP small | AP medium | AP large | AR@512 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mask R-CNN R50 (reference) | 0.2937 | 0.5927 | 0.2551 | 0.1637 | 0.3670 | 0.3958 | 0.4260 |
| Unified-Prompt SAM3 | 0.3642 | 0.7014 | 0.3331 | 0.2351 | 0.4257 | 0.5019 | 0.4768 |
| **Granularity-aware SAM3** | **0.3764** | **0.7235** | **0.3448** | **0.2415** | **0.4349** | **0.5555** | **0.4892** |
| Δ vs unified-prompt | +0.0122 | +0.0221 | +0.0117 | +0.0063 | +0.0092 | +0.0536 | +0.0124 |

## Boundary quality (instance-relative band, k = 0.19)

| | Bnd AP | Bnd AP50 | Bnd AP75 | Bnd small | Bnd medium | Bnd large | Uniform 8 px AP (comparison) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mask R-CNN R50 (reference) | 0.1337 | 0.4046 | 0.0545 | 0.0495 | 0.1713 | 0.2552 | 0.0869 |
| Unified-Prompt SAM3 | 0.1728 | 0.5096 | 0.0717 | 0.0816 | 0.2081 | 0.3340 | 0.1152 |
| **Granularity-aware SAM3** | **0.1766** | **0.5227** | 0.0703 | 0.0815 | 0.2127 | 0.3617 | 0.1219 |
| Δ vs unified-prompt | +0.0038 | +0.0131 | −0.0014 | −0.0001 | +0.0046 | +0.0276 | +0.0067 |

## Band-rule behaviour (Claim 1)

Boundary AP / Mask AP per crown-size bin. 1.0 means the band has collapsed into Mask AP;
near 0 means the band has floored. The usable window is approximately 0.15 to 0.75.

| band rule | small | medium | large | usable bins |
|---|---:|---:|---:|---|
| Image-relative (0.02 times image diagonal, approximately 58 px) | 1.00 | 1.00 | 0.99 | 0/3; approaches Mask AP |
| Uniform 8 px | 0.91 | 0.25 | 0.09 | 1/3; degenerate at both ends |
| **Instance-relative, k = 0.19** | **0.35** | **0.49** | **0.67** | **3/3 usable** |

The instance-relative band rule `d = k × equivalent_diameter` with k = 0.19 is the only rule
of the three that keeps every crown-size bin inside the usable window in the matched final
individual-tree evaluations for Unified-Prompt SAM3 and the Mask R-CNN reference. The value
was fixed before the final model comparison; exact preliminary calibration estimates are not
reported because that sweep combined different reporting populations.

## Pre-registered efficacy verdict (granularity-aware arm)

The decision rule was declared and hash-bound to the frozen Unified-Prompt SAM3 evaluation before the arm was
trained (`Context/active/prompt_granularity_efficacy.json`, SHA-256
`4e649c7060b514171f7407d631b7d0c5f7176c6ee5eb91d60db6073a62d65974`):

| criterion | role | threshold | observed Δ | verdict |
|---|---|---:|---:|---|
| Mask AP | primary effect | ≥ +0.0100 | **+0.012159** | **PASS** |
| Boundary AP, instance k=0.19 | non-regression | ≥ −0.0030 | **+0.003784** | **PASS** |
| Mask AP small | non-regression | ≥ −0.0030 | **+0.006349** | **PASS** |

## Interpretation limits

- One training seed (42). Deltas are descriptive; no standard deviation or training-seed variability estimate is reported.
- TEST informed in-training monitoring and is not an untouched model-selection holdout.
- The collapsed-category evaluation (trees and canopy groups scored as one class) moves in
  the opposite direction from the tree-only result, consistent with the intended
  prompt-conditioned granularity specialization rather than a generic prediction increase.

## Provenance

| Artifact | Identity |
|---|---|
| Reporting ground truth (TEST export) | SHA-256 `1e9cf5120bf3be17a61ccf800e48fe86b8581b7223cb20888bdfb89fed8d8f4b` |
| Unified-Prompt SAM3 evaluation record (internal A0) | SHA-256 `16dd053a4090a9dd34b209b63a8dd396819d0c600462a4d39fca99fdbde63635` |
| Granularity-aware final weights | 3,371,949,809 bytes, first-64-MiB SHA-256 `e2398e160469a63e742879c8504f7becec76fa04183666f3289cf69a67f516ef` |
| Pre-trained initialization | `facebook/sam3` checkpoint (`runtime/checkpoints/sam3/sam3.pt`) |

Every inference dump carries an `inference_config.json` recording weights identity, tiling,
precision, and merge settings; the Unified-Prompt SAM3 and granularity-aware dumps agree on all
prediction-affecting fields.
