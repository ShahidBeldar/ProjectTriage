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

| Metric | Value |
|---|---|
| Validation accuracy | 93.1% |
| Melanoma recall | 82.5% (up from 73% after class-weighted fine-tuning) |
| Inference time | ~15ms / image |
| ECE (calibration) | 0.11 |
| Uncertainty, correct predictions | 0.19 (avg) |
| Uncertainty, wrong predictions | 0.32 (avg) |

The gap between the last two rows is the core validation of this
approach: the model's uncertainty estimate is measurably higher when
it is wrong, which is what makes risk-based triage meaningful rather
than cosmetic.

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

training/           Kaggle notebooks (training record, not imported
                     by the running application)

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
