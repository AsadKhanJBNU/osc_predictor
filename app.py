"""
app.py — Predict organic solar cell donor-acceptor pair properties from SMILES alone.

Fully self-contained: no other project files needed besides best_models/ (trained
models + configs), Examined_Dataset.csv (optional, for reference), and requirements.txt.

Run:  python app.py
"""
import json
import numpy as np
import pandas as pd
import gradio as gr
import joblib
import spaces
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Draw
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# ------------------------------------------------------------------------------------
# SMILES -> fingerprint featurization (Morgan + Layered + AtomPair)
# ------------------------------------------------------------------------------------
NBITS = 1024

def to_arr(bitvect, nbits):
    arr = np.zeros((nbits,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr

def fp_morgan(mol):
    return to_arr(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=NBITS), NBITS)

def fp_layered(mol):
    return to_arr(Chem.LayeredFingerprint(mol, fpSize=NBITS), NBITS)

def fp_atompair(mol):
    return to_arr(rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=NBITS), NBITS)

FP_FUNCS = {'morgan': fp_morgan, 'layered': fp_layered, 'atompair': fp_atompair}


def build_fingerprints_for_column(smiles_series, fp_name, side_prefix):
    func = FP_FUNCS[fp_name]
    mols = [Chem.MolFromSmiles(s) for s in smiles_series]
    fps = np.vstack([func(m) if m is not None else np.zeros(NBITS, dtype=np.int8) for m in mols])
    cols = [f'{side_prefix}_{fp_name}_{i}' for i in range(NBITS)]
    return pd.DataFrame(fps, columns=cols, index=smiles_series.index)


def build_all_fingerprints(donor_smiles, acceptor_smiles):
    parts = []
    for fp_name in FP_FUNCS:
        parts.append(build_fingerprints_for_column(donor_smiles, fp_name, 'D'))
        parts.append(build_fingerprints_for_column(acceptor_smiles, fp_name, 'A'))
    return pd.concat(parts, axis=1)


def align_to_training_columns(X_new, training_feature_columns):
    return X_new.reindex(columns=training_feature_columns, fill_value=0)

ALL_TARGETS = ['PCE', 'V_OC', 'J_SC', 'HOMO_D', 'Lambda+', 'HOMO_A', 'LUMO_A', 'Lambda-']

UNITS = {
    'PCE': '%', 'V_OC': 'V', 'J_SC': 'mA/cm2',
    'HOMO_D': 'eV', 'Lambda+': 'eV', 'HOMO_A': 'eV', 'LUMO_A': 'eV', 'Lambda-': 'eV',
}
DESCRIPTIONS = {
    'PCE': 'Power Conversion Efficiency',
    'V_OC': 'Open-Circuit Voltage',
    'J_SC': 'Short-Circuit Current',
    'HOMO_D': 'Donor HOMO Energy',
    'Lambda+': 'Donor Reorganization Energy',
    'HOMO_A': 'Acceptor HOMO Energy',
    'LUMO_A': 'Acceptor LUMO Energy',
    'Lambda-': 'Acceptor Reorganization Energy',
}

MOL_IMG_SIZE = (340, 280)

# Three example donor/acceptor pairs, pulled directly from the dataset. Clicking a
# button below fills the two SMILES boxes directly with these values.
EXAMPLE_1 = ("CCc1ccsc1-c1cc(CC)c(-c2sccc2CC)s1",
             "CCC1c2ccccc2-c2cc3c4cccc5c4c(c3cc21)c1cccc2c3cc4c(cc3c5c21)C(CC)c1ccccc1-4")
EXAMPLE_2 = ("CCc1ccsc1-c1cc(CC)c(-c2sccc2CC)s1",
             "CCN1C(=O)c2ccc(C=Cc3ccc(C=Cc4ccc5c(c4)C(=O)N(CC)C5=O)c4nsnc34)cc2C1=O")
EXAMPLE_3 = ("CCc1ccsc1-c1cc(CC)c(-c2sccc2CC)s1",
             "C=c1c(=Cc2ccc(-c3cc4c(s3)-c3cc5c(cc3C4(CC)CC)-c3sc(-c4ccc(C=c6sc(=S)n(CC)c6=C)c6nsnc46)cc3C5(CC)CC)c3nsnc23)sc(=S)n1CC")


# ------------------------------------------------------------------------------------
# Model loading (cached after first use per target)
# ------------------------------------------------------------------------------------
_CACHE = {}

def _load_target(target):
    if target in _CACHE:
        return _CACHE[target]
    with open(f'best_models/{target}/run_config.json') as f:
        cfg = json.load(f)
    model = joblib.load(f"best_models/{cfg['best_model_file']}")
    _CACHE[target] = (cfg, model)
    return cfg, model


def _predict_with_model(loaded_model, X):
    """Handles both plain sklearn-style models and the custom stacking bundle
    ({'is_stack_bundle': True, 'base_models': ..., 'meta_model': ...})."""
    if isinstance(loaded_model, dict) and loaded_model.get('is_stack_bundle'):
        base_preds = np.column_stack([
            loaded_model['base_models'][n].predict(X.values) for n in loaded_model['top3_model_names']
        ])
        return loaded_model['meta_model'].predict(base_preds)[0]
    else:
        return loaded_model.predict(X)[0]


def _mol_image(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=MOL_IMG_SIZE)


# ------------------------------------------------------------------------------------
# Main prediction function
# ------------------------------------------------------------------------------------
@spaces.GPU
@spaces.GPU
def _zerogpu_startup_check():
    """
    Unused placeholder. This Space runs on ZeroGPU hardware, which requires at least
    one @spaces.GPU-decorated function to exist so HF's startup check passes. All real
    prediction work below runs as ordinary CPU Python (no GPU needed for sklearn/
    XGBoost/LightGBM/CatBoost inference) and does NOT go through this decorator, to
    avoid routing simple CPU calls through ZeroGPU's GPU-allocation subprocess.
    """
    return None


def predict_multi(donor_smiles, acceptor_smiles, selected_targets):
    donor_smiles = (donor_smiles or '').strip()
    acceptor_smiles = (acceptor_smiles or '').strip()
    empty_df = pd.DataFrame(columns=["Property", "Description", "Predicted Value", "Unit", "Model"])

    if not donor_smiles or not acceptor_smiles:
        return None, None, empty_df, "Please enter both donor and acceptor SMILES."

    donor_img = _mol_image(donor_smiles)
    if donor_img is None:
        return None, None, empty_df, "Donor SMILES could not be parsed. Please check it."

    acceptor_img = _mol_image(acceptor_smiles)
    if acceptor_img is None:
        return donor_img, None, empty_df, "Acceptor SMILES could not be parsed. Please check it."

    if not selected_targets:
        return donor_img, acceptor_img, empty_df, "Please select at least one property to predict."

    donor_series = pd.Series([donor_smiles])
    acceptor_series = pd.Series([acceptor_smiles])
    X_raw = build_all_fingerprints(donor_series, acceptor_series)

    rows = []
    for target in selected_targets:
        cfg, model = _load_target(target)
        X = align_to_training_columns(X_raw, cfg['feature_columns'])
        pred = _predict_with_model(model, X)
        rows.append({
            "Property": target,
            "Description": DESCRIPTIONS.get(target, ""),
            "Predicted Value": f"{pred:.4f}",
            "Unit": UNITS.get(target, ""),
            "Model": cfg['best_model_name'],
        })

    df = pd.DataFrame(rows)
    status = f"Prediction complete for {len(rows)} propert{'y' if len(rows) == 1 else 'ies'}."
    return donor_img, acceptor_img, df, status


def fill_example(example):
    return example[0], example[1]


# ------------------------------------------------------------------------------------
# UI — plain, minimal, no icons
# ------------------------------------------------------------------------------------
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
    gr.Markdown("# Organic Solar Cell Donor-Acceptor Property Predictor")
    gr.Markdown(
        "Predict electronic and photovoltaic properties of a donor-acceptor pair directly from "
        "molecular structure, without DFT calculation or device fabrication.",
        elem_id="subtitle",
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("Input structures", elem_classes="section-title")
            donor = gr.Textbox(label="Donor SMILES", placeholder="Enter donor SMILES")
            acceptor = gr.Textbox(label="Acceptor SMILES", placeholder="Enter acceptor SMILES")

            with gr.Row():
                ex1_btn = gr.Button("Example 1", elem_classes="example-btn", size="sm")
                ex2_btn = gr.Button("Example 2", elem_classes="example-btn", size="sm")
                ex3_btn = gr.Button("Example 3", elem_classes="example-btn", size="sm")

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
        Predictions are estimates from machine-learning models trained on a dataset of 319
        experimentally characterized donor-acceptor pairs, and should be treated as a
        pre-synthesis screening aid rather than a substitute for DFT calculation or experimental
        measurement. See the accompanying paper for per-property accuracy (R2) and a discussion
        of limitations.
        """
    )

if __name__ == "__main__":
    import os
    if os.environ.get("SPACE_ID"):
        demo.launch(theme=theme, css=CUSTOM_CSS)
    else:
        demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True,
                    theme=theme, css=CUSTOM_CSS)
