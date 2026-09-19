"""
Improved SMILES featurization for OSC donor/acceptor pairs.

The original pipeline used only 1024-bit Morgan + Layered + AtomPair fingerprints
(~5000 columns, 319 rows). That p >> n setting overfits, and it ignores
physicochemical properties that correlate with HOMO/LUMO/reorganization energy.

This module builds a compact, chemistry-informed feature set:
  - RDKit 2D descriptors (size, polarity, conjugation, topology, partial charge)
  - Heteroatom counts important for OSC materials (F, S, Se, Si, N, ...)
  - MACCS keys
  - Compact Morgan (r=2 and r=3) and AtomPair bit fingerprints
  - Pair-level interaction features (diffs, products, Tanimoto, XOR)

Still structure-only: no DFT energies and no device measurements as inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors
from rdkit.Chem import rdPartialCharges
from rdkit import DataStructs
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

NBITS = 256
MACCS_NBITS = 167

try:
    from rdkit.Chem import rdFingerprintGenerator

    _MORGAN2 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=NBITS)
    _MORGAN3 = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=NBITS)
    _ATOMPAIR = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=NBITS)

    def _fp_morgan(mol, radius=2):
        gen = _MORGAN2 if radius == 2 else _MORGAN3
        return np.asarray(gen.GetFingerprint(mol), dtype=np.int8)

    def _fp_atompair(mol):
        return np.asarray(_ATOMPAIR.GetFingerprint(mol), dtype=np.int8)

except Exception:
    def _fp_morgan(mol, radius=2):
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=NBITS)
        arr = np.zeros((NBITS,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(bv, arr)
        return arr

    def _fp_atompair(mol):
        bv = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=NBITS)
        arr = np.zeros((NBITS,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(bv, arr)
        return arr


ELEMENT_Z = {
    "nF": 9, "nCl": 17, "nBr": 35, "nI": 53,
    "nS": 16, "nN": 7, "nO": 8, "nSe": 34, "nSi": 14, "nB": 5,
}

SCALAR_DESCRIPTORS = [
    ("MolWt", Descriptors.MolWt),
    ("ExactMolWt", Descriptors.ExactMolWt),
    ("HeavyAtomCount", Descriptors.HeavyAtomCount),
    ("NumHAcceptors", Descriptors.NumHAcceptors),
    ("NumHDonors", Descriptors.NumHDonors),
    ("NumRotatableBonds", Descriptors.NumRotatableBonds),
    ("NumAromaticRings", Descriptors.NumAromaticRings),
    ("NumAliphaticRings", Descriptors.NumAliphaticRings),
    ("NumSaturatedRings", Descriptors.NumSaturatedRings),
    ("RingCount", Descriptors.RingCount),
    ("NumAromaticHeterocycles", Descriptors.NumAromaticHeterocycles),
    ("NumAromaticCarbocycles", Descriptors.NumAromaticCarbocycles),
    ("NumHeteroatoms", Descriptors.NumHeteroatoms),
    ("TPSA", Descriptors.TPSA),
    ("MolLogP", Descriptors.MolLogP),
    ("MolMR", Descriptors.MolMR),
    ("FractionCSP3", Descriptors.FractionCSP3),
    ("BertzCT", Descriptors.BertzCT),
    ("HallKierAlpha", Descriptors.HallKierAlpha),
    ("Kappa1", Descriptors.Kappa1),
    ("Kappa2", Descriptors.Kappa2),
    ("Kappa3", Descriptors.Kappa3),
    ("Chi0v", Descriptors.Chi0v),
    ("Chi1v", Descriptors.Chi1v),
    ("Chi2v", Descriptors.Chi2v),
    ("Chi0n", Descriptors.Chi0n),
    ("Chi1n", Descriptors.Chi1n),
    ("LabuteASA", Descriptors.LabuteASA),
    ("BalabanJ", Descriptors.BalabanJ),
    ("MaxEStateIndex", Descriptors.MaxEStateIndex),
    ("MinEStateIndex", Descriptors.MinEStateIndex),
    ("MaxAbsEStateIndex", Descriptors.MaxAbsEStateIndex),
    ("qed", Descriptors.qed),
    ("NumValenceElectrons", Descriptors.NumValenceElectrons),
    ("NHOHCount", Descriptors.NHOHCount),
    ("NOCount", Descriptors.NOCount),
]


def smiles_to_mol(smiles: str):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles)


def _safe_desc(fn, mol, default=0.0):
    try:
        val = fn(mol)
        if val is None or not np.isfinite(val):
            return default
        return float(val)
    except Exception:
        return default


def _partial_charge_stats(mol):
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
        charges = []
        for atom in mol.GetAtoms():
            try:
                q = float(atom.GetDoubleProp("_GasteigerCharge"))
            except Exception:
                continue
            if np.isfinite(q):
                charges.append(q)
        if not charges:
            return 0.0, 0.0, 0.0, 0.0
        arr = np.asarray(charges, dtype=float)
        return float(arr.max()), float(arr.min()), float(np.abs(arr).max()), float(arr.std())
    except Exception:
        return 0.0, 0.0, 0.0, 0.0


def mol_scalar_features(mol) -> dict:
    """Physicochemical / electronic descriptors for one molecule."""
    feats = {}
    if mol is None:
        names = [n for n, _ in SCALAR_DESCRIPTORS]
        extra = [
            "nF", "nCl", "nBr", "nI", "nS", "nN", "nO", "nSe", "nSi", "nB",
            "nAromaticAtoms", "nConjugatedBonds", "aromaticAtomFrac",
            "tpsa_over_mw", "MaxPartialCharge", "MinPartialCharge",
            "MaxAbsPartialCharge", "PartialChargeStd", "sps",
        ]
        return {k: 0.0 for k in names + extra}

    for name, fn in SCALAR_DESCRIPTORS:
        feats[name] = _safe_desc(fn, mol)

    counts = {k: 0 for k in ELEMENT_Z}
    n_aromatic_atoms = 0
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        for key, az in ELEMENT_Z.items():
            if z == az:
                counts[key] += 1
        if atom.GetIsAromatic():
            n_aromatic_atoms += 1
    feats.update({k: float(v) for k, v in counts.items()})
    feats["nAromaticAtoms"] = float(n_aromatic_atoms)

    n_conj = 0
    for bond in mol.GetBonds():
        try:
            if bond.GetIsConjugated():
                n_conj += 1
        except Exception:
            pass
    feats["nConjugatedBonds"] = float(n_conj)

    hac = feats.get("HeavyAtomCount", 0.0) or 0.0
    feats["aromaticAtomFrac"] = float(n_aromatic_atoms / hac) if hac else 0.0
    mw = feats.get("MolWt", 0.0) or 0.0
    feats["tpsa_over_mw"] = float(feats.get("TPSA", 0.0) / mw) if mw else 0.0

    mx, mn, amx, std = _partial_charge_stats(mol)
    feats["MaxPartialCharge"] = mx
    feats["MinPartialCharge"] = mn
    feats["MaxAbsPartialCharge"] = amx
    feats["PartialChargeStd"] = std

    sps_fn = getattr(Descriptors, "SPS", None)
    feats["sps"] = _safe_desc(sps_fn, mol) if sps_fn is not None else 0.0
    return feats


def mol_fingerprints(mol) -> dict:
    """Compact fingerprints for one molecule. Returns 1-d numpy arrays."""
    if mol is None:
        return {
            "maccs": np.zeros(MACCS_NBITS, dtype=np.int8),
            "morgan2": np.zeros(NBITS, dtype=np.int8),
            "atompair": np.zeros(NBITS, dtype=np.int8),
            "layered": np.zeros(NBITS, dtype=np.int8),
        }
    maccs = np.zeros(MACCS_NBITS, dtype=np.int8)
    try:
        bv = MACCSkeys.GenMACCSKeys(mol)
        DataStructs.ConvertToNumpyArray(bv, maccs)
    except Exception:
        pass
    layered = np.zeros(NBITS, dtype=np.int8)
    try:
        bv = Chem.LayeredFingerprint(mol, fpSize=NBITS)
        DataStructs.ConvertToNumpyArray(bv, layered)
    except Exception:
        pass
    return {
        "maccs": maccs,
        "morgan2": _fp_morgan(mol, 2),
        "atompair": _fp_atompair(mol),
        "layered": layered,
    }


def tanimoto(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.dot(a, b))
    union = float(a.sum() + b.sum() - inter)
    return inter / union if union else 0.0


class MolFeatureCache:
    def __init__(self):
        self._store = {}

    def get(self, smiles: str):
        if smiles in self._store:
            return self._store[smiles]
        mol = smiles_to_mol(smiles)
        packed = {
            "mol": mol,
            "scalar": mol_scalar_features(mol),
            "fp": mol_fingerprints(mol),
        }
        self._store[smiles] = packed
        return packed


def _prefix_scalar(d: dict, prefix: str) -> dict:
    return {f"{prefix}{k}": v for k, v in d.items()}


def _fp_frame(fp_dict: dict, prefix: str, index) -> pd.DataFrame:
    parts = []
    for name, arr in fp_dict.items():
        cols = [f"{prefix}{name}_{i}" for i in range(len(arr))]
        parts.append(pd.DataFrame([arr], columns=cols, index=index))
    return pd.concat(parts, axis=1)


def build_molecule_features(smiles_series: pd.Series, prefix: str, cache: MolFeatureCache) -> pd.DataFrame:
    """Feature matrix for a single-molecule target (donor-only or acceptor-only)."""
    scalar_rows = []
    fp_rows = []
    for smi in smiles_series:
        packed = cache.get(smi)
        scalar_rows.append(_prefix_scalar(packed["scalar"], prefix))
        fp_rows.append(_fp_frame(packed["fp"], prefix, [0]).iloc[0])
    scalars = pd.DataFrame(scalar_rows).reset_index(drop=True)
    fps = pd.DataFrame(fp_rows).reset_index(drop=True)
    return pd.concat([scalars, fps], axis=1)


PROD_KEYS = [
    "MolWt", "TPSA", "MolLogP", "NumAromaticRings", "BertzCT",
    "nConjugatedBonds", "nF", "nS", "nSe", "HeavyAtomCount",
    "aromaticAtomFrac", "MaxAbsPartialCharge",
]


def build_pair_features(donor_smiles: pd.Series, acceptor_smiles: pd.Series,
                        cache: MolFeatureCache) -> pd.DataFrame:
    """Donor + acceptor descriptors/fingerprints plus interaction features."""
    d_scalar, a_scalar = [], []
    d_fp, a_fp = [], []
    extra = []
    for d_smi, a_smi in zip(donor_smiles, acceptor_smiles):
        d = cache.get(d_smi)
        a = cache.get(a_smi)
        d_scalar.append(_prefix_scalar(d["scalar"], "D_"))
        a_scalar.append(_prefix_scalar(a["scalar"], "A_"))
        d_fp.append(_fp_frame(d["fp"], "D_", [0]).iloc[0])
        a_fp.append(_fp_frame(a["fp"], "A_", [0]).iloc[0])

        row = {
            "DA_tanimoto_morgan2": tanimoto(d["fp"]["morgan2"], a["fp"]["morgan2"]),
            "DA_tanimoto_maccs": tanimoto(d["fp"]["maccs"], a["fp"]["maccs"]),
        }
        xor = np.bitwise_xor(d["fp"]["morgan2"].astype(np.int8), a["fp"]["morgan2"].astype(np.int8))
        both = np.bitwise_and(d["fp"]["morgan2"].astype(np.int8), a["fp"]["morgan2"].astype(np.int8))
        row["DA_morgan2_xor_sum"] = float(xor.sum())
        row["DA_morgan2_and_sum"] = float(both.sum())
        for k in d["scalar"]:
            dv, av = d["scalar"][k], a["scalar"][k]
            row[f"DA_diff_{k}"] = float(dv - av)
            row[f"DA_absdiff_{k}"] = float(abs(dv - av))
            if k in PROD_KEYS:
                row[f"DA_prod_{k}"] = float(dv * av)
        extra.append(row)

    out = pd.concat([
        pd.DataFrame(d_scalar).reset_index(drop=True),
        pd.DataFrame(a_scalar).reset_index(drop=True),
        pd.DataFrame(d_fp).reset_index(drop=True),
        pd.DataFrame(a_fp).reset_index(drop=True),
        pd.DataFrame(extra).reset_index(drop=True),
    ], axis=1)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def align_to_training_columns(X_new: pd.DataFrame, training_feature_columns) -> pd.DataFrame:
    return X_new.reindex(columns=training_feature_columns, fill_value=0)
