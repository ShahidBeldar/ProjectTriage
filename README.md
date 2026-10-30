# AI-Based Skin Lesion Detection and Uncertainty-Aware Risk Triage

B.Tech final-year project combining a Swin Transformer visual backbone,
clinical metadata, Cross-Attention multimodal fusion, and Evidential
Deep Learning (EDL) for uncertainty-aware skin lesion classification.

Unlike a standard classifier, this system also estimates how confident
it is in each prediction, and uses that uncertainty — along with an
automated image quality check — to assign a risk level (Low Risk /
Suspicious / Expert Review) rather than a single flat label.

## Architecture

```
Dermoscopy Image + Clinical Metadata (age, sex, location)
            |
            v
   Image Quality Pre-screening
            |
   +--------+--------+
   |                 |
Swin Transformer   Metadata Encoder
(visual features)  (schema-driven)
   |                 |
   +--------+--------+
            v
  Cross-Attention Fusion
            v
    Evidential Deep Learning (EDL)
            v
  Prediction + Uncertainty
            v
     Risk Triage (3-tier)
```

## Results

All figures below come from `training/final_evaluation.ipynb`, run on the
final class-weighted checkpoint over the **1,521-image validation split**
(patient-level split; 0 image overlap and 0 lesion overlap with the 8,494
training images). There is no separate held-out test set, so these are
validation results.

### Classification

| Metric | Value |
|---|---|
| Validation accuracy | 93.4% (1,421 / 1,521) |
| Melanoma recall | 82.1% (up from 73% after class-weighted fine-tuning) |
| Melanoma precision | 74.3% |
| Macro F1 | 0.92 |

| Class | Precision | Recall | Support |
|---|---|---|---|
| akiec | 0.907 | 0.929 | 42 |
| bcc | 0.934 | 1.000 | 57 |
| bkl | 0.916 | 0.882 | 161 |
| df | 0.909 | 0.952 | 21 |
| mel | 0.743 | 0.821 | 162 |
| nv | 0.970 | 0.955 | 1059 |
| vasc | 1.000 | 1.000 | 19 |

### Uncertainty and calibration

| Metric | Value |
|---|---|
| Uncertainty, correct predictions | 0.19 (mean) |
| Uncertainty, wrong predictions | 0.39 (mean) |
| Error-detection AUROC | 0.87 |
| ECE (10 bins) | 0.125 (model is mildly under-confident) |

The gap between the two uncertainty rows is the core validation of this
approach: the model's uncertainty is measurably higher when it is wrong,
which is what makes risk-based triage meaningful rather than cosmetic.

### Risk-triage tiers (by uncertainty, quality check aside)

| Tier | Share of cases | Accuracy | Share of all errors |
|---|---|---|---|
| Low (u <= 0.25) | 78.2% | 97.7% | 28% |
| Moderate (0.25 < u <= 0.45) | 13.3% | 84.7% | 31% |
| Expert review (u > 0.45) | 8.5% | 68.2% | 41% |

Routing the 21.8% of cases with u > 0.25 to closer review captures 72% of
the model's mistakes. 156 of 162 melanomas (96.3%) are not classed as
"confident benign".

### Inference time

| Setting | Time |
|---|---|
| Single image (batch size 1), model forward pass, GPU (T4) | 34.7 ms (std 2.4) |
| Batched (batch size 16), per image | 13.5 ms |

### Robustness and quality check

Measured on a fixed random sample of 500 validation images, each
degraded at increasing severity.

| Degradation | Level | Accuracy | Mean uncertainty | Flagged by quality check | Sent to Expert Review |
|---|---|---|---|---|---|
| none (clean) | - | 92.6% | 0.208 | 2.6% | 16.8% |
| blur | r=1 / 2 / 4 / 8 | 91.0 / 83.8 / 71.2 / 62.2% | 0.207 / 0.216 / 0.246 / 0.233 | 24 / 92 / 100 / 100% | 31 / 94 / 100 / 100% |
| dark | x0.5 / 0.25 / 0.1 | 89.6 / 73.6 / 59.0% | 0.195 / 0.148 / 0.445 | 60 / 96 / 100% | 65 / 96 / 100% |
| bright | x1.5 / 2 / 3 | 80.2 / 43.6 / 26.0% | 0.265 / 0.273 / 0.297 | 31 / 79 / 96% | 40 / 83 / 97% |
| low-res | 96 / 64 / 32 px | 84.6 / 74.2 / 67.4% | 0.220 / 0.249 / 0.231 | 100% | 100% |
| noise | sigma 15 / 40 / 80 | 88.4 / 75.0 / 69.4% | 0.226 / 0.251 / 0.279 | 0% | 14 / 13 / 12% |

Takeaways: accuracy falls steadily as images degrade, and the quality
check catches blur, darkness, strong over-exposure and low resolution.
Mean uncertainty alone does *not* rise reliably with degradation, so the
quality pre-screen and the uncertainty estimate are complementary: the
pre-screen handles unusable inputs, the evidential uncertainty handles
ambiguous but usable ones. Sensor noise is not detected (see limitations).

## How the metrics are calculated

- **Split.** HAM10000 (10,015 images, 7 classes) is split at patient/lesion
  level into 8,494 train and 1,521 validation images. All metrics use the
  validation images, evaluated in eval mode with no augmentation
  (resize to 224 x 224, ImageNet normalisation).
- **Accuracy and recall.** Predicted class = argmax of the Dirichlet mean.
  Accuracy = correct / total; per-class recall = correct in class / class
  support (melanoma recall uses the `mel` row of the confusion matrix).
- **Evidential uncertainty.** The head outputs non-negative evidence e_k
  per class. With alpha_k = e_k + 1 and S = sum(alpha_k), the class
  probability is p_k = alpha_k / S and the uncertainty is u = K / S with
  K = 7. Little evidence gives a large u.
- **Calibration (ECE).** Predictions are grouped into 10 equal-width bins by
  confidence (max p_k); ECE = sum over bins of (n_bin / N) x |accuracy_bin
  - mean confidence_bin|. Lower is better.
- **Error-detection AUROC.** Uncertainty is used as the score for "this
  prediction is wrong" (wrong = positive class); 0.5 is chance, 1.0 is
  perfect.
- **Triage tiers.** Cases are bucketed by u with the thresholds in
  `core/risk_triage.py` (0.25 and 0.45). "Share of all errors" is the
  fraction of all misclassifications that land in that tier.
- **Inference time.** Timed around the model forward pass only (image
  already preprocessed), with GPU synchronisation, after warm-up.
  Single-image: 95 timed runs at batch size 1 (first 5 discarded).
  Batched: mean per-image time over the whole validation set at batch size
  16. Image decoding, preprocessing and the quality check are not included.
- **Quality check.** `core/quality_check.py` flags an image if the shorter
  side is below 100 px, the mean grey level is below 25 or above 230, or
  the variance of the Laplacian (on a 256 px thumbnail) is below 21.7.
  The blur threshold was calibrated on 800 *training* images (2nd
  percentile of their blur scores). The earlier value of 50 flagged 35.8%
  of clean validation images; 21.7 flags 3.0% (46 / 1,521).
- **Robustness test.** 500 fixed random validation images are degraded
  (Gaussian blur radius 1-8, brightness x0.1-x3, downscaling to 96-32 px on
  the short side, Gaussian pixel noise sigma 15-80) and passed through the
  quality check, model and triage. "Sent to Expert Review" counts images
  that fail the quality check or have u > 0.45.

## Reproducing the evaluation

Run `training/final_evaluation.ipynb` on Kaggle with the HAM10000 dataset,
the training notebook output (`multimodal_edl_model_weighted.pth`,
`val_ids.csv`, `val_metadata.csv`) and the `core/` folder attached.

## Project Structure

```
core/               Shared, environment-agnostic logic
  architecture.py    Model classes (Swin wrapper, fusion, EDL head)
  metadata_schema.py Schema-driven clinical metadata encoder
  risk_triage.py     3-tier risk logic
  quality_check.py   Image quality pre-screening
  env.py             Kaggle-vs-local path resolution

api/                FastAPI backend
  main.py            Routes
  predictor.py       Wraps core/ into a single predict() call
  tests/

web/                React frontend (Vite)

training/           Kaggle notebooks (training record and
                     final_evaluation.ipynb; not imported by the app)

models/             Local model storage (gitignored; empty on GitHub)

docs/               Methodology and evaluation writeups
```

## Running Locally

1. Place `multimodal_edl_model_weighted.pth` and `metadata_encoder.pkl`
   in `models/`.
2. `pip install -r requirements.txt`
3. `python -m api.main` — backend runs at `http://localhost:8000`,
   docs at `http://localhost:8000/docs`.
4. `cd web && npm install && npm run dev` — frontend at
   `http://localhost:5173`.

## Running on Kaggle

`core/env.py` auto-detects the Kaggle environment (it checks for
`/kaggle/input`) and resolves model files from any attached dataset —
no path changes needed. Attach a dataset containing the two files
above, import the `core`/`api` modules, and run the FastAPI app with
`pyngrok` for a public URL (see `training/` notebooks for the exact
cells used).

## Known Limitations

- No separate held-out test set: all reported numbers are on the
  validation split, which was also used to select the checkpoint.
- Sensor noise and mild over-exposure (brightness x1.5) are not caught by
  the quality pre-screen, and mean uncertainty does not rise reliably on
  degraded images. Out-of-distribution inputs (non-dermoscopy photos) are
  not explicitly handled.
- ECE of 0.125 means the model is mildly under-confident; no post-hoc
  calibration is applied.
- Quality-check thresholds are calibrated on HAM10000 dermoscopy images
  and may need re-tuning for other devices.
- Trained on a single source (HAM10000); accuracy degrades on
  out-of-distribution images from different acquisition devices
  (domain shift). Multi-source training is future work.
- `skin_color`, `evolution`, `symptoms`, and `previous_disease_history`
  are defined in the metadata schema as extension points but are not
  yet trained — HAM10000 has no labels for them. See
  `core/metadata_schema.py`.

## Disclaimer

Academic research prototype. Not a diagnostic device. Always consult
a qualified dermatologist.
