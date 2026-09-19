# OSC Donor–Acceptor Property Predictor

Predicts **nine** electronic and photovoltaic properties of organic solar cell
donor–acceptor pairs **directly from SMILES** — no DFT and no device measurements
as inputs.

**Try it live:** https://huggingface.co/spaces/AsadWazir/osc_predictor

## About

Selected models from the combined SMILES-only pipeline:

| Property | Description | Selected model | Test *R*² |
|---|---|---|---|
| V_OC | Open-circuit voltage | ElasticNet | 45.8% |
| J_SC | Short-circuit current | ElasticNet | 69.5% |
| PCE | Power conversion efficiency | Ridge | 61.2% |
| HOMO_D | Donor HOMO | k-NN | 18.5% |
| LUMO_D | Donor LUMO | Extra Trees | 39.1% |
| HOMO_A | Acceptor HOMO | k-NN | 52.5% |
| LUMO_A | Acceptor LUMO | ElasticNet | 30.8% |
| Lambda+ | Donor reorganization energy | Extra Trees | 43.0% |
| Lambda- | Acceptor reorganization energy | Extra Trees | 63.8% |

Features are RDKit 2D descriptors, MACCS keys, compact Morgan / AtomPair / Layered
fingerprints (256 bits), and donor–acceptor interaction terms. Device targets are
trained on 2,296 pairs after physical-range filtering of a 2,356-pair literature
corpus. Orbital models use unique-molecule electrochemical labels; λ+ / λ− use
the DFT-labelled Padula–Troisi subset only.

On a 35-pair experimental hold-out (SMILES-pair identity excluded from training),
PCE *R*² = 60.5% (MAE = 2.80%).

## Repository contents

```
.
├── Dataset/                 # Padula–Troisi + OPV-DB + 35-pair hold-out
├── best_models/             # selected model + config for each of 9 targets
├── app.py                   # Gradio predictor
├── utils_featurization.py   # SMILES-only feature builder (must match training)
└── requirements.txt
```

## How to run

```bash
git clone https://github.com/AsadKhanJBNU/osc_predictor.git
cd osc_predictor
pip install -r requirements.txt
python app.py
```

Opens a local Gradio app at `http://127.0.0.1:7860`. Example pairs from the
paper hold-out (PDTBTBO:ITIC, PSFTZ:Y6, D18:BTP-Th) are filled by the buttons.

## Requirements

See `requirements.txt`. Core dependencies:

- **RDKit** — descriptors and fingerprints from SMILES
- **scikit-learn, joblib** — Ridge, ElasticNet, k-NN, Extra Trees inference
- **Gradio** — web interface
- **pandas, numpy** — tables and arrays

Pin **scikit-learn** to the version in `requirements.txt`. Newer or older
sklearn builds often cannot load the saved `.joblib` files.

## Dataset

| File | Contents |
|---|---|
| `Examined_Dataset.csv` | 319 Padula–Troisi pairs (device metrics experimental; orbitals/λ DFT) |
| `Examined_Dataset_1.csv` | OPV-DB literature pairs (device + electrochemical orbitals) |
| `external_test_set.csv` | 35-pair experimental hold-out |
| `Combined_Dataset.csv` | Combined table with split labels |


## Notes and limitations

- Use the models to **screen** donor–acceptor combinations, not to replace
  device fabrication.
- Predictions shrink toward the mean: low values are over-predicted and high
  values are under-predicted. Extreme PCE / *J*_SC estimates are rankings, not
  calibrated device forecasts.
- Hold-out exclusion is at the **pair** SMILES level. A donor or acceptor may
  still appear in training with a different partner.
- HOMO_D is the weakest target (*R*² = 18.5% on the modelling test split).
- λ+ / λ− have no labels on the 35-pair hold-out.
