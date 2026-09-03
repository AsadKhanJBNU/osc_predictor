# OSC Donor-Acceptor Property Predictor

Predicts electronic and photovoltaic properties of organic solar cell donor-acceptor pairs
**directly from molecular structure (SMILES)** — no DFT calculation or device fabrication
required.

**Try it live:** https://huggingface.co/spaces/AsadWazir/osc_predictor

## About

This repository provides a trained machine-learning framework that predicts eight properties
of a donor-acceptor pair from just the donor and acceptor SMILES strings:

| Property | Description |
|---|---|
| PCE | Power Conversion Efficiency |
| V_OC | Open-Circuit Voltage |
| J_SC | Short-Circuit Current |
| HOMO_D | Donor HOMO Energy |
| Lambda+ | Donor Reorganization Energy |
| HOMO_A | Acceptor HOMO Energy |
| LUMO_A | Acceptor LUMO Energy |
| Lambda- | Acceptor Reorganization Energy |

Molecular fingerprints (Morgan + Layered + AtomPair) are computed from SMILES and used with
ensemble tree models (CatBoost, XGBoost, LightGBM, Random Forest, Extra Trees, Gradient
Boosting, HistGradientBoosting, and a stacking ensemble of the best-performing models) —
fine-tuned per property on a dataset of 319 experimentally characterized donor-acceptor pairs.

## Repository contents

```
.
├── d.csv     # the 319-pair training dataset
├── best_models/             # trained model + config for each of the 8 properties
│   ├── PCE/
│   ├── V_OC/
│   ├── J_SC/
│   ├── HOMO_D/
│   ├── Lambda+/
│   ├── HOMO_A/
│   ├── LUMO_A/
│   └── Lambda-/
├── app.py                   # self-contained prediction app (no other .py files needed)
└── requirements.txt
```

## How to run

```bash
git clone https://github.com/AsadKhanJBNU/osc_predictor.git
cd osc_predictor
pip install -r requirements.txt
python app.py
```

This launches a local web interface (Gradio) at `http://127.0.0.1:7860`. Enter a donor SMILES
and an acceptor SMILES, select which of the 8 properties to predict, and click **Predict**.
Example donor-acceptor pairs are provided as quick-fill buttons in the interface.

## Requirements

See `requirements.txt` for exact pinned versions. Core dependencies:

- **RDKit** — molecular fingerprint generation from SMILES
- **scikit-learn, XGBoost, LightGBM, CatBoost** — the trained model backends
- **Gradio** — the web interface
- **pandas, numpy, joblib, scipy** — supporting data/model handling

Library versions in `requirements.txt` are pinned to match those used during model training.
Installing different (especially newer) versions of scikit-learn, XGBoost, LightGBM, or
CatBoost than what's listed **may cause the saved models to fail to load**, since these
libraries do not always guarantee backward compatibility for pickled/serialized models across
versions. If you hit a loading error, first confirm your installed versions exactly match
`requirements.txt`.

## Dataset

`Examined_Dataset.csv` contains 319 experimentally characterized donor-acceptor pairs (262
distinct donors, 76 distinct acceptors), originally compiled by Padula and Troisi (2019),
*Advanced Energy Materials*, ["Concurrent Optimization of Organic Donor-Acceptor Pairs through
Machine Learning"](https://doi.org/10.1002/aenm.201902463). Please cite the original source if
you use this dataset independently of this repository.

## Notes and limitations

- Predictions are estimates from models trained on a relatively small (319-pair) experimental
  dataset, and should be treated as a **pre-synthesis screening aid**, not a substitute for DFT
  calculation or experimental measurement.
- Accuracy varies substantially by property — donor-only and acceptor-only electronic
  properties (e.g. HOMO_D) are generally predicted more reliably than device-level properties
  (e.g. V_OC), which depend on factors beyond molecular structure alone (morphology,
  processing conditions). See the accompanying paper for detailed per-property accuracy (R²)
  and discussion.
