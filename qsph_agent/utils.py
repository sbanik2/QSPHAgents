from __future__ import annotations

from typing import Optional, Dict, List, Literal, Any
import os
import warnings
import json

import numpy as np
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder


# =========================
# Config utilities
# =========================

def load_keys_and_config(config_path: str = "config.yaml"):
    """
    Load API keys from .env and global config from YAML.

    Side effects:
        - Sets os.environ["OPENAI_API_KEY"] if present in .env.

    Returns:
        MP_API_KEY (str | None): Materials Project API key.
        CONFIG (dict): Parsed YAML configuration (empty dict if not found).
    """
    load_dotenv()

    openai_key = os.getenv("OPENAI_API_KEY")
    mp_key = os.getenv("MP_API_KEY")

    # Ensure OpenAI SDK sees the key
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}
        print(f"[Warning] Config file '{config_path}' not found. Using empty CONFIG.")

    return mp_key, cfg


# =========================
# DOS hypothesis schema
# =========================

class PeakInfo(BaseModel):
    positions_eV: List[float] = Field(..., description="Peak positions in eV")
    intensities: List[float] = Field(..., description="Peak intensities (normalized or raw)")


class DOSHypothesis(BaseModel):
    # Primary qualitative + quantitative attributes
    material_classification: Literal[
        "metallic",
        "semiconducting",
        "insulating",
        "half-metallic",
        "pseudogapped",
    ]
    overall_dos_shape: Optional[str] = Field(
        None,
        description="General shape like flat, peaked, V-shaped, U-shaped, etc.",
    )
    asymmetry_comment: Optional[str] = Field(
        None,
        description="Symmetry or skewness between valence and conduction bands",
    )
    valence_band_peaks: Optional[PeakInfo]
    conduction_band_peaks: Optional[PeakInfo]
    pseudogap_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="0 = flat metallic DOS, 1 = deep full gap, ~0.5 = partial dip",
    )

    # Reasoning per field
    reasoning: Optional[Dict[str, str]] = Field(
        None,
        description=(
            "Dict of reasoning or justification for each field. "
            "Keys should match attribute names."
        ),
    )


def dos_hypothesis_to_dict(hypothesis: DOSHypothesis) -> dict:
    """
    Convert a DOSHypothesis object into a dictionary format.
    Includes all fields and their reasoning if available.
    """

    def serialize_peaks(peaks: Optional[Any]) -> Optional[dict]:
        if peaks is None:
            return None
        try:
            return {
                "positions_eV": [round(p, 4) for p in peaks.positions_eV],
                "intensities": [round(i, 4) for i in peaks.intensities],
            }
        except Exception as e:
            return {"error": f"Invalid peak format: {str(e)}"}

    return {
        "material_classification": hypothesis.material_classification,
        "overall_dos_shape": hypothesis.overall_dos_shape,
        "asymmetry_comment": hypothesis.asymmetry_comment,
        "valence_band_peaks": serialize_peaks(hypothesis.valence_band_peaks),
        "conduction_band_peaks": serialize_peaks(hypothesis.conduction_band_peaks),
        "pseudogap_score": hypothesis.pseudogap_score,
        "reasoning": hypothesis.reasoning or {},
    }


def dos_hypothesis_to_description(hypothesis: DOSHypothesis) -> str:
    """
    Convert a DOSHypothesis object into a human-readable, formatted description string.
    Includes reasoning text where available.
    """

    def format_peaks(peaks: Optional[Any], label: str) -> str:
        if peaks is None:
            return f"{label}: Not specified"
        try:
            positions = ", ".join(f"{p:.2f} eV" for p in peaks.positions_eV)
            intensities = ", ".join(f"{i:.2f}" for i in peaks.intensities)
            return (
                f"{label} peaks at energies: [{positions}], "
                f"with intensities: [{intensities}]"
            )
        except Exception as e:
            return f"{label}: Invalid peak format ({e})"

    R = hypothesis.reasoning or {}

    lines = [
        f"**Material Classification**: {hypothesis.material_classification}",
        f"  ↳ Reasoning: {R.get('material_classification')}",
        "",
        f"**Overall DOS Shape**: {hypothesis.overall_dos_shape or 'Not specified'}",
        f"  ↳ Reasoning: {R.get('overall_dos_shape')}",
        "",
        f"**Asymmetry Comment**: {hypothesis.asymmetry_comment or 'Not specified'}",
        f"  ↳ Reasoning: {R.get('asymmetry_comment')}",
        "",
        format_peaks(hypothesis.valence_band_peaks, "Valence Band"),
        f"  ↳ Reasoning: {R.get('valence_band_peaks')}",
        "",
        format_peaks(hypothesis.conduction_band_peaks, "Conduction Band"),
        f"  ↳ Reasoning: {R.get('conduction_band_peaks')}",
        "",
        f"**Pseudogap Score**: {hypothesis.pseudogap_score}",
        f"  ↳ Reasoning: {R.get('pseudogap_score')}",
    ]

    # Drop empty lines
    return "\n".join(line for line in lines if line.strip())


# =========================
# SimpleDOSPredictor
# =========================

class SimpleDOSPredictor:
    """Quantitative estimates of DOS parameters using KNN on structural descriptors."""

    def __init__(self, n_neighbors: int = 4):
        self.n_neighbors = n_neighbors
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.models: Dict[str, Any] = {}

    def _extract_features(self, struct_dict: Dict[str, Any]) -> np.ndarray:
        feats = struct_dict.get("structural_features", {})
        return np.array(
            [
                feats.get("space_group_number", 0) or 0,
                feats.get("volume_per_atom", 0.0) or 0.0,
                feats.get("density", 0.0) or 0.0,
                feats.get("valence_electron_count", 0.0) or 0.0,
                feats.get("avg_coordination_number", 0.0) or 0.0,
                feats.get("mean_bond_length", 0.0) or 0.0,
                feats.get("bond_length_std", 0.0) or 0.0,
                feats.get("electronegativity_mean", 0.0) or 0.0,
                feats.get("electronegativity_difference", 0.0) or 0.0,
            ],
            dtype=float,
        )

    def _extract_targets(self, dos_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant target fields from DOS; keep None for missing values."""
        vb = dos_dict.get("valence_band_peaks") or {}
        cb = dos_dict.get("conduction_band_peaks") or {}

        return {
            "material_classification": dos_dict.get("material_classification", None),
            "overall_dos_shape": dos_dict.get("overall_dos_shape", None),
            "pseudogap_score": dos_dict.get("pseudogap_score", None),
            "valence_main_energy": vb.get("main_peak_energy", None),
            "valence_main_height": vb.get("main_peak_height", None),
            "conduction_main_energy": cb.get("main_peak_energy", None),
            "conduction_main_height": cb.get("main_peak_height", None),
        }

    def fit(self, dataset: list):
        """
        Fit KNN classifiers/regressors on a list of entries of the form:
            {
                "structure": {...},
                "dos": {
                    "dos_description_dict": {...}
                }
            }
        """
        X = []
        Y_all = []

        for i, entry in enumerate(dataset):
            try:
                struct_feats = self._extract_features(entry["structure"])
                dos_dict = entry.get("dos", {}).get("dos_description_dict", {})
                targets = self._extract_targets(dos_dict)

                if np.isnan(struct_feats).any():
                    warnings.warn(
                        f"NaN encountered in features for entry {i}; skipping."
                    )
                    continue

                # Require at least the main labels for classification
                if not targets["material_classification"] or not targets["overall_dos_shape"]:
                    warnings.warn(
                        f"Missing classification labels for entry {i}; skipping."
                    )
                    continue

                X.append(struct_feats)
                Y_all.append(targets)

            except Exception as e:
                warnings.warn(f"Skipping entry {i} due to error: {str(e)}")
                continue

        if not X:
            raise ValueError("No valid training data found.")

        X = np.array(X)
        X_scaled = self.scaler.fit_transform(X)
        Y_dict = {key: [y[key] for y in Y_all] for key in Y_all[0].keys()}

        # === Classification ===
        for field in ["material_classification", "overall_dos_shape"]:
            values = Y_dict[field]
            le = LabelEncoder()
            y_enc = le.fit_transform(values)
            clf = KNeighborsClassifier(n_neighbors=self.n_neighbors)
            clf.fit(X_scaled, y_enc)
            self.models[field] = clf
            self.label_encoders[field] = le

        # === Regression ===
        reg_fields = [
            "pseudogap_score",
            "valence_main_energy",
            "valence_main_height",
            "conduction_main_energy",
            "conduction_main_height",
        ]

        for field in reg_fields:
            raw_vals = Y_dict[field]
            # Replace None with np.nan before casting
            y = np.array(
                [v if v is not None else np.nan for v in raw_vals],
                dtype=np.float32,
            )
            mask = ~np.isnan(y)
            if mask.sum() < self.n_neighbors:
                warnings.warn(
                    f"Skipping regression model for '{field}' "
                    f"due to insufficient non-NaN data ({mask.sum()} samples)."
                )
                continue

            reg = KNeighborsRegressor(n_neighbors=self.n_neighbors)
            reg.fit(X_scaled[mask], y[mask])
            self.models[field] = reg

    def predict(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict DOS-related quantities from a structure dict of the form:
            {"structural_features": {...}, ...}
        """
        x = self._extract_features(structure).reshape(1, -1)

        if np.isnan(x).any():
            warnings.warn(
                "NaN encountered in input structure features. Prediction aborted."
            )
            return {}

        x_scaled = self.scaler.transform(x)
        result: Dict[str, Any] = {}

        # === Classification ===
        for field in ["material_classification", "overall_dos_shape"]:
            clf = self.models.get(field)
            le = self.label_encoders.get(field)
            if clf is None or le is None:
                result[field] = None
                continue

            try:
                pred = clf.predict(x_scaled)
                result[field] = le.inverse_transform(pred)[0]
            except Exception as e:
                warnings.warn(f"Classification failed for '{field}': {str(e)}")
                result[field] = None

        # === Regression ===
        reg_fields = [
            "pseudogap_score",
            "valence_main_energy",
            "valence_main_height",
            "conduction_main_energy",
            "conduction_main_height",
        ]

        for field in reg_fields:
            reg = self.models.get(field)
            if reg is None:
                result[field] = None
                continue

            try:
                pred = float(reg.predict(x_scaled)[0])
                if field == "pseudogap_score":
                    pred = max(0.0, min(1.0, pred))  # clamp to [0, 1]
                result[field] = pred
            except Exception as e:
                warnings.warn(f"Regression failed for '{field}': {str(e)}")
                result[field] = None

        # === Reassemble composite outputs ===
        result["valence_band_peaks"] = {
            "main_peak_energy": result.pop("valence_main_energy", None),
            "main_peak_height": result.pop("valence_main_height", None),
        }
        result["conduction_band_peaks"] = {
            "main_peak_energy": result.pop("conduction_main_energy", None),
            "main_peak_height": result.pop("conduction_main_height", None),
        }

        return result


# =========================
# Critique schema
# =========================

class DOSCritique(BaseModel):
    key_disagreements: List[str] = Field(
        ...,
        description=(
            "List of key inconsistencies between the hypothesis and the "
            "quantitative / structural information."
        ),
    )
    suggestions: List[str] = Field(
        ...,
        description="Suggestions to improve or correct the DOS hypothesis.",
    )
    summary: Optional[str] = Field(
        None,
        description="High-level qualitative comment from the DOS critic.",
    )
