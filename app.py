"""
Predict OSC donor–acceptor properties from SMILES only.

Nine targets, structure-only features (RDKit descriptors, MACCS, compact
fingerprints, pair interactions). No DFT and no V_OC/J_SC as inputs.

Run:  python app.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger

from utils_featurization import (
    MolFeatureCache,
    align_to_training_columns,
    build_molecule_features,
    build_pair_features,
)

RDLogger.DisableLog("rdApp.*")

try:
    import spaces
except ImportError:  # local run without Hugging Face
    class spaces:  # type: ignore
        @staticmethod
        def GPU(fn):
            return fn

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "best_models"

PAIR_TARGETS = ["V_OC", "J_SC", "PCE"]
DONOR_TARGETS = ["HOMO_D", "LUMO_D", "Lambda+"]
ACCEPTOR_TARGETS = ["HOMO_A", "LUMO_A", "Lambda-"]
ALL_TARGETS = PAIR_TARGETS + DONOR_TARGETS + ACCEPTOR_TARGETS

UNITS = {
    "V_OC": "V",
    "J_SC": "mA cm⁻²",
    "PCE": "%",
    "HOMO_D": "eV",
    "LUMO_D": "eV",
    "HOMO_A": "eV",
    "LUMO_A": "eV",
    "Lambda+": "eV",
    "Lambda-": "eV",
}
DESCRIPTIONS = {
    "V_OC": "Open-circuit voltage",
    "J_SC": "Short-circuit current density",
    "PCE": "Power conversion efficiency",
    "HOMO_D": "Donor HOMO energy",
    "LUMO_D": "Donor LUMO energy",
    "HOMO_A": "Acceptor HOMO energy",
    "LUMO_A": "Acceptor LUMO energy",
    "Lambda+": "Donor reorganization energy (λ+)",
    "Lambda-": "Acceptor reorganization energy (λ−)",
}

# Hold-out examples used in the paper (PDTBTBO:ITIC, PSFTZ:Y6, D18:BTP-Th).
EXAMPLE_1 = (
    "CCCCCCC(CCCC)COc1c(OCC(CCCC)CCCCCC)c(-c2ccc(-c3ccc(C)s3)s2)c2nsnc2c1C",
    "CCCCCCc1ccc(C2(c3ccc(CCCCCC)cc3)c3cc4c(cc3-c3sc5cc(/C=C6\\C(=O)c7ccccc7C6=C(C#N)C#N)sc5c32)C(c2ccc(CCCCCC)cc2)(c2ccc(CCCCCC)cc2)c2c-4sc3cc(/C=C4\\C(=O)c5ccccc5C4=C(C#N)C#N)sc23)cc1",
)
EXAMPLE_2 = (
    "CCCCCCCCc1cc(-c2nnc(-c3cc(CCCCCCCC)c(-c4cc5c(-c6cc(F)c(SCC(CCCC)CCCCCC)s6)c6sc(C)cc6c(-c6cc(F)c(SCC(CCCC)CCCCCC)s6)c5s4)s3)nn2)sc1C",
    "CCCCCCCCCCCc1c(/C=C2\\C(=O)c3cc(F)c(F)cc3C2=C(C#N)C#N)sc2c1sc1c3c4nsnc4c4c5sc6c(CCCCCCCCCCC)c(/C=C7\\C(=O)c8cc(F)c(F)cc8C7=C(C#N)C#N)sc6c5n(CC(CC)CCCC)c4c3n(CC(CC)CCCC)c21",
)
EXAMPLE_3 = (
    "CCCCCCC(CCCC)Cc1csc(-c2cc3c4nsnc4c4cc(-c5cc(CC(CCCC)CCCCCC)c(-c6cc7c(-c8cc(F)c(CC(CC)CCCC)s8)c8sccc8c(-c8cc(F)c(CC(CC)CCCC)s8)c7s6)s5)sc4c3s2)c1",
    "CCCCCCc1ccc(-c2c(/C=C3\\C(=O)c4cc(F)c(F)cc4C3=C(C#N)C#N)sc3c2sc2c4c5nsnc5c5c6sc7c(-c8ccc(CCCCCC)s8)c(/C=C8\\C(=O)c9cc(F)c(F)cc9C8=C(C#N)C#N)sc7c6n(CC(CCCC)CCCCCC)c5c4n(CC(CCCC)CCCCCC)c32)s1",
)

MOL_IMG_SIZE = (340, 280)
_CACHE = {}
_FEAT_CACHE = MolFeatureCache()


def _load_target(target):
    if target in _CACHE:
        return _CACHE[target]
    cfg_path = MODEL_DIR / target / "run_config.json"
    model_path = MODEL_DIR / target / "best_model.joblib"
    with cfg_path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    model = joblib.load(model_path)
    _CACHE[target] = (cfg, model)
    return cfg, model


def _predict_one(model, X):
    import numpy as np

    Xv = np.asarray(X, dtype=float)
    if hasattr(X, "to_numpy"):
        Xv = X.to_numpy(dtype=float, copy=False)
    if isinstance(model, dict) and model.get("is_stack_bundle"):
        base = np.column_stack(
            [model["base_models"][n].predict(Xv) for n in model["top3_model_names"]]
        )
        pred = model["meta_model"].predict(base)
    else:
        pred = model.predict(Xv)
    return float(pred[0])


def _mol_image(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=MOL_IMG_SIZE)


def _features_for(target, donor, acceptor):
    d = pd.Series([donor])
    a = pd.Series([acceptor])
    if target in PAIR_TARGETS:
        return build_pair_features(d, a, _FEAT_CACHE)
    if target in DONOR_TARGETS:
        return build_molecule_features(d, "D_", _FEAT_CACHE)
    return build_molecule_features(a, "A_", _FEAT_CACHE)


@spaces.GPU
def _zerogpu_startup_check():
    """Placeholder so Hugging Face ZeroGPU Spaces pass the GPU-function check."""
    return None


def predict_multi(donor_smiles, acceptor_smiles, selected_targets):
    donor_smiles = (donor_smiles or "").strip()
    acceptor_smiles = (acceptor_smiles or "").strip()
    empty = pd.DataFrame(columns=["Property", "Description", "Predicted Value", "Unit", "Model"])

    if not donor_smiles or not acceptor_smiles:
        return None, None, empty, "Please enter both donor and acceptor SMILES."

    donor_img = _mol_image(donor_smiles)
    if donor_img is None:
        return None, None, empty, "Donor SMILES could not be parsed. Please check it."

    acceptor_img = _mol_image(acceptor_smiles)
    if acceptor_img is None:
        return donor_img, None, empty, "Acceptor SMILES could not be parsed. Please check it."

    if not selected_targets:
        return donor_img, acceptor_img, empty, "Please select at least one property to predict."

    rows = []
    for target in selected_targets:
        cfg, model = _load_target(target)
        X = align_to_training_columns(_features_for(target, donor_smiles, acceptor_smiles), cfg["feature_columns"])
        pred = _predict_one(model, X)
        rows.append({
            "Property": target,
            "Description": DESCRIPTIONS.get(target, ""),
            "Predicted Value": f"{pred:.4f}",
            "Unit": UNITS.get(target, ""),
            "Model": cfg.get("best_model_name", ""),
        })

    df = pd.DataFrame(rows)
    status = f"Prediction complete for {len(rows)} propert{'y' if len(rows) == 1 else 'ies'}."
    return donor_img, acceptor_img, df, status


def fill_example(example):
    return example[0], example[1]


CUSTOM_CSS = """
.gradio-container {max-width: 980px !important; margin: auto; font-family: 'Inter', -apple-system, sans-serif;}
h1 {font-weight: 600 !important; font-size: 1.6em !important; margin-bottom: 0.1em !important;}
#subtitle {color: #555; margin-bottom: 1.6em; font-size: 0.98em; line-height: 1.5;}
#predict-btn {height: 2.8em; font-size: 1em; font-weight: 500;}
.section-title {font-size: 0.82em; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em;
                color: #888; margin-top: 0.6em; margin-bottom: 0.4em; border-bottom: 1px solid #eee;
                padding-bottom: 0.4em;}
.example-btn {font-size: 0.85em !important; padding: 0.4em 0.9em !important; height: auto !important;}
footer {visibility: hidden}
"""

theme = gr.themes.Default(
    primary_hue="slate",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
).set(
    button_primary_background_fill="#1f2937",
    button_primary_background_fill_hover="#374151",
    button_primary_text_color="#ffffff",
    block_border_width="1px",
    block_shadow="none",
)

with gr.Blocks(title="OSC Property Predictor") as demo:
    gr.Markdown("# Organic Solar Cell Donor–Acceptor Property Predictor")
    gr.Markdown(
        "Predict nine electronic and photovoltaic properties of a donor–acceptor pair "
        "from SMILES only — no DFT and no device measurements as inputs.",
        elem_id="subtitle",
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("Input structures", elem_classes="section-title")
            donor = gr.Textbox(label="Donor SMILES", placeholder="Enter donor SMILES")
            acceptor = gr.Textbox(label="Acceptor SMILES", placeholder="Enter acceptor SMILES")

            with gr.Row():
                ex1_btn = gr.Button("PDTBTBO:ITIC", elem_classes="example-btn", size="sm")
                ex2_btn = gr.Button("PSFTZ:Y6", elem_classes="example-btn", size="sm")
                ex3_btn = gr.Button("D18:BTP-Th", elem_classes="example-btn", size="sm")

            gr.Markdown("Properties to predict", elem_classes="section-title")
            targets = gr.CheckboxGroup(
                choices=ALL_TARGETS, value=ALL_TARGETS, label=None, show_label=False,
            )

            predict_btn = gr.Button("Predict", variant="primary", elem_id="predict-btn")
            status = gr.Markdown()

        with gr.Column(scale=1):
            gr.Markdown("Molecular structures", elem_classes="section-title")
            with gr.Row():
                donor_img_out = gr.Image(label="Donor", height=260, show_label=True)
                acceptor_img_out = gr.Image(label="Acceptor", height=260, show_label=True)

    gr.Markdown("Predicted properties", elem_classes="section-title")
    results_table = gr.Dataframe(
        headers=["Property", "Description", "Predicted Value", "Unit", "Model"],
        interactive=False, wrap=True,
    )

    ex1_btn.click(lambda: fill_example(EXAMPLE_1), outputs=[donor, acceptor])
    ex2_btn.click(lambda: fill_example(EXAMPLE_2), outputs=[donor, acceptor])
    ex3_btn.click(lambda: fill_example(EXAMPLE_3), outputs=[donor, acceptor])

    predict_btn.click(
        predict_multi,
        inputs=[donor, acceptor, targets],
        outputs=[donor_img_out, acceptor_img_out, results_table, status],
    )

    gr.Markdown(
        """
        ---
        Models were trained on 2,356 literature donor–acceptor pairs with SMILES-only
        features. Device targets use regularized linear models (Ridge / ElasticNet);
        orbital and reorganization-energy targets use k-NN or Extra Trees. Predictions
        are a pre-synthesis screening aid, not a substitute for fabrication. See the
        accompanying paper for per-property *R*², residual shrinkage, and the 35-pair
        experimental hold-out.
        """
    )

if __name__ == "__main__":
    if os.environ.get("SPACE_ID"):
        demo.launch(theme=theme, css=CUSTOM_CSS)
    else:
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            show_error=True,
            theme=theme,
            css=CUSTOM_CSS,
        )
