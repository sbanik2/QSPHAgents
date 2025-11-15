import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from pymatgen.electronic_structure.dos import Dos, Spin


# -------- DEFAULT DOS CONFIG (used only as fallback) --------

DEFAULT_DOS_CFG = {
    "gap_source": "metadata",
    "smoothing": 1.0,
    "metal_threshold": 0.05,
    "gap_threshold": 0.05,
    "sc_threshold": 5.0,
    "pseudogap_window": [-0.3, 0.3],
    "conduction_ref_window": [0.5, 2.0],
    "asymmetry_left": [-1.0, 0.0],
    "asymmetry_right": [0.0, 1.0],
    "valence_peaks_window": [-6.0, 0.0],
    "conduction_peaks_window": [0.0, 6.0],
    "sc_gap_span": 0.1,
    "sc_peak_threshold": 5.0,
}


# -------- utility ----------------


def window_mask(E, low, high):
    """Create a boolean mask for energy values between low and high."""
    return (E >= low) & (E <= high)


def finite_diff(y, x):
    """Calculate first and second derivatives of y with respect to x."""
    dy = np.gradient(y, x)   # First derivative (slope)
    d2y = np.gradient(dy, x)  # Second derivative (curvature)
    return dy, d2y


def find_peaks_summary(E, dos, energy_range, prominence=0.05):
    """
    Find prominent peaks in the DOS within specified energy range.

    Returns:
        list of dicts with keys: energy, height, prominence, width
    """
    mask = window_mask(E, energy_range[0], energy_range[1])
    if np.sum(mask) < 5:
        return []

    E_range = E[mask]
    dos_range = dos[mask]

    min_prominence = prominence * (np.max(dos_range) - np.min(dos_range) + 1e-12)

    peak_indices, peak_properties = find_peaks(dos_range, prominence=min_prominence)

    peaks = []
    if len(peak_indices) == 0:
        return peaks

    for i in range(len(peak_indices)):
        idx = peak_indices[i]
        peak_energy = float(E_range[idx])
        peak_height = float(dos_range[idx])
        peak_prominence = float(peak_properties["prominences"][i])

        if "widths" in peak_properties:
            width_in_samples = float(peak_properties["widths"][i])
            energy_per_sample = (E_range[-1] - E_range[0]) / len(E_range)
            fwhm = width_in_samples * energy_per_sample
        else:
            fwhm = np.nan

        peaks.append(
            {
                "energy": peak_energy,
                "height": peak_height,
                "prominence": peak_prominence,
                "width": fwhm,
            }
        )

    peaks.sort(key=lambda x: x["prominence"], reverse=True)
    return peaks[:3]


def detect_superconducting_gap(E, dos, gap_span=0.1, peak_threshold=5.0):
    """
    Detect signature of a superconducting gap.

    Heuristic:
    - Very low DOS at EF within [-gap_span, +gap_span]
    - Finite DOS a bit away from EF (typical_dos)
    - Clear "edges" of the gap where DOS rises again
    - Coherence peaks near gap edges that are much higher than typical_dos

    Returns:
        is_superconductor (bool)
        gap_width (float)
        peak_ratio (float)
    """
    # Window around EF for the gap
    gap_mask = window_mask(E, -gap_span, gap_span)
    if not np.any(gap_mask):
        return False, 0.0, 0.0

    E_gap = E[gap_mask]
    dos_gap = dos[gap_mask]

    # Make sure the EF (0.0) lies within the sampled gap window
    if not (E_gap[0] <= 0.0 <= E_gap[-1]):
        return False, 0.0, 0.0

    # DOS at EF
    dos_at_ef = float(np.interp(0.0, E_gap, dos_gap))
    if dos_at_ef > 1e-3:
        # Too much DOS at EF -> unlikely to be a clean SC gap
        return False, 0.0, 0.0

    # Region just outside the gap to define a "typical" DOS level
    far_mask_left = window_mask(E, -2 * gap_span, -gap_span)
    far_mask_right = window_mask(E, gap_span, 2 * gap_span)
    far_mask = far_mask_left | far_mask_right
    if not np.any(far_mask):
        return False, 0.0, 0.0

    typical_dos = float(np.median(dos[far_mask]))

    # If typical_dos is essentially zero, don't trust the SC heuristic
    if typical_dos <= 1e-4:
        return False, 0.0, 0.0

    # Find where DOS "recovers" towards typical_dos to define gap edges
    edge_threshold = 0.5 * typical_dos
    above_threshold = dos_gap > edge_threshold
    if not np.any(above_threshold):
        return False, 0.0, 0.0

    edge_indices = np.where(above_threshold)[0]
    if len(edge_indices) < 2:
        return False, 0.0, 0.0

    left_edge = float(E_gap[edge_indices[0]])
    right_edge = float(E_gap[edge_indices[-1]])
    gap_width = right_edge - left_edge
    if gap_width <= 0:
        return False, 0.0, 0.0

    # Look for coherence peaks near the gap edges (slightly wider windows)
    edge_window = 0.02  # eV, can be made configurable if you want
    left_peak_mask = window_mask(E, left_edge - edge_window, left_edge + edge_window)
    right_peak_mask = window_mask(E, right_edge - edge_window, right_edge + edge_window)

    left_peak = float(np.max(dos[left_peak_mask])) if np.any(left_peak_mask) else 0.0
    right_peak = float(np.max(dos[right_peak_mask])) if np.any(right_peak_mask) else 0.0
    avg_peak = 0.5 * (left_peak + right_peak)

    if typical_dos <= 0:
        peak_ratio = 0.0
    else:
        peak_ratio = avg_peak / typical_dos
        if not np.isfinite(peak_ratio):
            peak_ratio = 0.0

    is_superconductor = (peak_ratio > peak_threshold) and (gap_width > 0)

    return is_superconductor, gap_width, peak_ratio



# ---------- MAIN FEATURE EXTRACTION (shape, pseudogap, peaks) ----------


def extract_dos_features(E_rel, dos, config: dict | None = None):
    """
    Extract DOS shape features, assuming E_rel is already aligned so EF = 0.

    config:
        Global configuration dictionary. This function will use:
            config["dos_features"]  (if present)
        or, if you pass only the DOS section:
            config  (treated directly as the dos config).

    Returns:
        dict with keys:
            N_EF, slope_EF, curvature_EF, pseudogap_score, asymmetry,
            is_superconductor, sc_gap_width, sc_peak_ratio,
            valence_peaks, conduction_peaks
    """
    cfg_global = config or {}
    # Allow both full global config and direct dos section
    dos_cfg = cfg_global.get("dos_features", cfg_global)
    cfg = {**DEFAULT_DOS_CFG, **dos_cfg}

    smoothing = cfg["smoothing"]
    pg_low, pg_high = cfg["pseudogap_window"]
    cond_low, cond_high = cfg["conduction_ref_window"]
    asym_l_low, asym_l_high = cfg["asymmetry_left"]
    asym_r_low, asym_r_high = cfg["asymmetry_right"]
    valence_win = cfg["valence_peaks_window"]
    conduction_win = cfg["conduction_peaks_window"]
    sc_gap_span = cfg["sc_gap_span"]
    sc_peak_threshold = cfg["sc_peak_threshold"]
    # New: minimum DOS level required to trust pseudogap metric
    pseudogap_min_ref = cfg.get("pseudogap_min_ref", 1e-3)

    # Apply slight smoothing to reduce noise
    if smoothing > 0:
        dos_smooth = gaussian_filter1d(dos, smoothing)
    else:
        dos_smooth = dos.copy()

    # Derivatives
    slope, curvature = finite_diff(dos_smooth, E_rel)

    # DOS at EF
    N_EF = float(np.interp(0.0, E_rel, dos_smooth))

    # Slope and curvature at EF
    slope_EF = float(np.interp(0.0, E_rel, slope))
    curvature_EF = float(np.interp(0.0, E_rel, curvature))

    # Pseudogap: compare DOS near EF vs typical conduction DOS
    near_ef = window_mask(E_rel, pg_low, pg_high)
    conduction = window_mask(E_rel, cond_low, cond_high)

    min_near_ef = float(np.min(dos_smooth[near_ef])) if np.any(near_ef) else np.nan
    median_conduction = (
        float(np.median(dos_smooth[conduction])) if np.any(conduction) else np.nan
    )

    # Only compute pseudogap_score if conduction DOS is meaningfully non-zero
    if (
        not np.isnan(min_near_ef)
        and not np.isnan(median_conduction)
        and median_conduction > pseudogap_min_ref
    ):
        pseudogap_score = min_near_ef / (median_conduction + 1e-12)
    else:
        # In deep insulators / very low DOS cases, pseudogap is not well-defined
        pseudogap_score = np.nan

    # Asymmetry around EF
    left_side = window_mask(E_rel, asym_l_low, asym_l_high)
    right_side = window_mask(E_rel, asym_r_low, asym_r_high)

    left_avg = float(np.mean(dos_smooth[left_side])) if np.any(left_side) else np.nan
    right_avg = float(np.mean(dos_smooth[right_side])) if np.any(right_side) else np.nan

    if not np.isnan(left_avg) and not np.isnan(right_avg):
        asymmetry = (right_avg - left_avg) / (right_avg + left_avg + 1e-12)
    else:
        asymmetry = np.nan

    # Superconducting gap heuristic
    is_sc, sc_gap_width, sc_peak_ratio = detect_superconducting_gap(
        E_rel,
        dos_smooth,
        gap_span=sc_gap_span,
        peak_threshold=sc_peak_threshold,
    )

    # Peaks
    valence_peaks = find_peaks_summary(E_rel, dos_smooth, valence_win)
    conduction_peaks = find_peaks_summary(E_rel, dos_smooth, conduction_win)

    return {
        "N_EF": N_EF,
        "slope_EF": slope_EF,
        "curvature_EF": curvature_EF,
        "pseudogap_score": pseudogap_score,
        "asymmetry": asymmetry,
        "is_superconductor": is_sc,
        "sc_gap_width": sc_gap_width,
        "sc_peak_ratio": sc_peak_ratio,
        "valence_peaks": valence_peaks,
        "conduction_peaks": conduction_peaks,
    }


def generate_description(features, config: dict | None = None):
    """
    Generate structured description from DOS features.

    config:
        Global configuration dictionary. This function will use:
            config["dos_features"]  (if present)
        or, if you pass only the DOS section:
            config  (treated directly as the dos config).

    Returns:
        description_dict (dict)
        description_text (str)
    """
    cfg_global = config or {}
    dos_cfg = cfg_global.get("dos_features", cfg_global)
    cfg = {**DEFAULT_DOS_CFG, **dos_cfg}

    metal_threshold = cfg["metal_threshold"]
    gap_threshold = cfg["gap_threshold"]
    sc_threshold = cfg["sc_threshold"]

    # Optional extra knobs
    insulator_threshold = cfg.get("insulator_threshold", 2.0)          # eV
    nef_warning_threshold = cfg.get("nef_warning_threshold", metal_threshold)

    band_gap = float(features["band_gap"])
    N_EF = float(features["N_EF"])

    description_parts = []
    desc_dict = {}

    # -----------------------------
    # 1. Material type (pure-system logic)
    # -----------------------------
    if features["is_superconductor"] and features["sc_peak_ratio"] > sc_threshold:
        # Superconducting branch
        gap_meV = features["sc_gap_width"] * 1000
        main_type = "superconducting"
        type_text = (
            f"Superconducting with gap of {gap_meV:.1f} meV and coherence peaks "
            f"({features['sc_peak_ratio']:.1f}× normal DOS)."
        )

    elif band_gap > gap_threshold:
        # --- GAPPED SYSTEMS: use band_gap to classify, ignore N_EF for type ---
        if band_gap >= insulator_threshold:
            main_type = "insulating"
            base_text = (
                f"Band insulator with a band gap of {band_gap:.2f} eV "
                f"(VBM: {features['VBM']:+.2f} eV, CBM: {features['CBM']:+.2f} eV)."
            )
        else:
            main_type = "semiconducting"
            base_text = (
                f"Semiconducting with a band gap of {band_gap:.2f} eV "
                f"(VBM: {features['VBM']:+.2f} eV, CBM: {features['CBM']:+.2f} eV)."
            )

        # Only add a *note* if DOS(EF) is suspiciously large, but do not change class
        if N_EF > nef_warning_threshold:
            note = (
                f" Note: the calculated DOS shows a finite N(E_F) = {N_EF:.2f} states/eV, "
                "which is likely due to numerical smearing or Fermi-level placement "
                "rather than actual doping in a pure compound."
            )
        else:
            note = ""

        type_text = base_text + note

    else:
        # --- SMALL-GAP / GAPLESS: classify by DOS at EF ---
        if N_EF >= metal_threshold:
            main_type = "metallic"
            type_text = f"Metallic with high DOS at EF: {N_EF:.2f} states/eV."
        else:
            main_type = "pseudogap"
            type_text = (
                "Pseudogap-like behaviour: low but finite DOS at the Fermi level "
                f"({N_EF:.3f} states/eV) without a well-developed insulating gap."
            )

    description_parts.append(type_text)
    desc_dict["material_classification"] = main_type

    # -----------------------------
    # 2. Shape near EF
    # -----------------------------
    if features["curvature_EF"] < -0.1:
        shape_text = "V-shaped suppression near EF."
    elif features["curvature_EF"] > 0.1:
        shape_text = "U-shaped DOS around EF."
    else:
        shape_text = "Relatively flat DOS near EF."

    description_parts.append(shape_text)
    desc_dict["overall_dos_shape"] = shape_text

    # -----------------------------
    # 3. Asymmetry
    # -----------------------------
    asymmetry_text = None
    if not np.isnan(features["asymmetry"]):
        if features["asymmetry"] > 0.2:
            asymmetry_text = (
                "Conduction side (0–1 eV) has higher DOS than valence (−1–0 eV)."
            )
        elif features["asymmetry"] < -0.2:
            asymmetry_text = (
                "Valence side (−1–0 eV) has higher DOS than conduction (0–1 eV)."
            )

    if asymmetry_text:
        description_parts.append(asymmetry_text)
    desc_dict["asymmetry_comment"] = asymmetry_text

    # -----------------------------
    # 4. Peak descriptions
    # -----------------------------
    def describe_peaks(peaks):
        if not peaks:
            return None
        main_peak = peaks[0]
        return {
            "main_peak_energy": main_peak["energy"],
            "main_peak_height": main_peak["height"],
            "other_peaks": [p["energy"] for p in peaks[1:]],
        }

    desc_dict["valence_band_peaks"] = describe_peaks(features["valence_peaks"])
    desc_dict["conduction_band_peaks"] = describe_peaks(features["conduction_peaks"])

    def peaks_to_text(peaks, band):
        if not peaks:
            return None
        text = f"{band} band peak at {peaks['main_peak_energy']:+.2f} eV"
        if peaks["other_peaks"]:
            text += (
                " with additional features at "
                + ", ".join(f"{e:+.2f}" for e in peaks["other_peaks"])
                + " eV."
            )
        else:
            text += "."
        return text

    valence_text = peaks_to_text(desc_dict["valence_band_peaks"], "Valence")
    conduction_text = peaks_to_text(desc_dict["conduction_band_peaks"], "Conduction")

    if valence_text:
        description_parts.append(valence_text)
    if conduction_text:
        description_parts.append(conduction_text)

    # -----------------------------
    # 5. Pseudogap score
    # -----------------------------
    if not np.isnan(features["pseudogap_score"]):
        pseudo_text = f"Pseudogap score: {features['pseudogap_score']:.2f}."
        description_parts.append(pseudo_text)
        desc_dict["pseudogap_score"] = features["pseudogap_score"]

    return desc_dict, " ".join(description_parts)


# ---------- Pymatgen integration + top-level API ----------


def _mp_dict_to_pmg_dos(mp_dos_data) -> Dos:
    """
    Convert a Materials Project DOS dict to a pymatgen Dos object.
    """
    energies = np.array(mp_dos_data["energies"])
    densities = mp_dos_data["densities"]
    efermi = float(mp_dos_data["efermi"])

    if isinstance(densities, dict):
        pmg_densities = {Spin(int(k)): np.array(v) for k, v in densities.items()}
    else:
        pmg_densities = {Spin.up: np.array(densities)}

    return Dos(efermi=efermi, energies=energies, densities=pmg_densities)


def get_dos_features(material_doc: dict, config: dict | None = None):
    """
    Process a full Materials Project record and return extracted features + description.

    Args:
        material_doc (dict):
            {
                "metadata": {...},   # includes band_gap, etc.
                "dos": {...},        # energies, densities, efermi, ...
                "cif": "..."
            }
        config (dict | None):
            Global config loaded centrally. This function will use:
                config["dos_features"] (if present)
            or, if you pass only the dos section:
                config (treated directly as the dos config).

    Returns:
        tuple:
            features (dict | None)
            description_dict (dict | None)
            description (str | None)
    """
    try:
        cfg_global = config or {}
        dos_cfg = cfg_global.get("dos_features", cfg_global)
        cfg = {**DEFAULT_DOS_CFG, **dos_cfg}

        metadata = material_doc.get("metadata", {})
        mp_dos_data = material_doc.get("dos", None)

        if mp_dos_data is None:
            return None, None, "ERROR processing DOS: no DOS data present"

        # Band gap comes from metadata (band-structure-based), robust to None / bad types
        gap_source = cfg.get("gap_source", "metadata")
        if gap_source == "metadata":
            bg_val = metadata.get("band_gap", 0.0)
            if bg_val is None:
                band_gap = 0.0
            else:
                try:
                    band_gap = float(bg_val)
                except (TypeError, ValueError):
                    band_gap = 0.0
        else:
            band_gap = 0.0  # placeholder if you later support DOS-based gap

        # Build pymatgen Dos object
        pmg_dos = _mp_dict_to_pmg_dos(mp_dos_data)

        energies = pmg_dos.energies
        efermi = pmg_dos.efermi

        # Total DOS (spin-summed or single-spin, handled internally by pymatgen)
        total_dos = np.array(pmg_dos.get_densities(spin=None), dtype=float)

        # Energy relative to EF (EF -> 0)
        E_rel = energies - efermi

        # Shape / pseudogap / peaks / SC features
        core_feats = extract_dos_features(E_rel, total_dos, config=cfg)

        # Simple VBM/CBM placeholders relative to EF for description
        if band_gap > 0:
            VBM = 0.0
            CBM = band_gap
        else:
            VBM = 0.0
            CBM = 0.0

        # Merge into single feature dict
        features = {
            "band_gap": band_gap,
            "VBM": VBM,
            "CBM": CBM,
        }
        features.update(core_feats)

        # Generate structured description
        desc_dict, description = generate_description(features, config=cfg)

        return features, desc_dict, description

    except Exception as error:
        return None, None, f"ERROR processing DOS: {error}"
