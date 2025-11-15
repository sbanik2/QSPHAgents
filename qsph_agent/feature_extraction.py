from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.periodic_table import Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np


def extract_material_features(
    parser,
    cutoff: float | None = None,
    primitive: bool | None = None,
    config: dict | None = None,
):
    """
    Extract structural features relevant to DOS interpretation.

    Args:
        parser: A pymatgen CifParser (or similar) object.
        cutoff (float | None): Neighbor cutoff distance. If None, will use
                               config["database"]["cutoff"] or 3.0 as fallback.
        primitive (bool | None): Whether to use primitive structure. If None,
                                 will use config["database"]["primitive"] or False.
        config (dict | None): Global configuration dictionary, typically loaded
                              from YAML. Example:

                              {
                                  "database": {
                                      "cutoff": 5.0,
                                      "primitive": true
                                  },
                                  ...
                              }

    Returns:
        dict: {
            "formula": str,
            "pymatgen_structure": dict,
            "structural_features": { ... }
        }
    """
    # ---- resolve config-driven parameters ----
    cfg_global = config or {}
    db_cfg = cfg_global.get("database", {})

    if cutoff is None:
        cutoff = db_cfg.get("cutoff", 3.0)

    if primitive is None:
        primitive = db_cfg.get("primitive", False)

    # ---- parse structure ----
    structure = parser.parse_structures(primitive=primitive)[0]

    # Symmetry
    sga = SpacegroupAnalyzer(structure)
    sg_symbol = sga.get_space_group_symbol()
    sg_number = sga.get_space_group_number()
    crystal_system = sga.get_crystal_system()

    # Volume/density
    volume_per_atom = structure.volume / len(structure.sites)
    density = structure.density

    # Coordination number and bond stats
    cnn = CrystalNN()
    cn_list = []
    bond_lengths = []

    for i, site in enumerate(structure.sites):
        try:
            cn = cnn.get_cn(structure, i)
        except Exception:
            cn = None
        cn_list.append(cn if cn is not None else np.nan)

        neighbors = structure.get_neighbors(site, cutoff)
        for n in neighbors:
            try:
                d = n[1]  # (site, dist) tuple
            except Exception:
                d = getattr(n, "distance", None)
            if d is not None:
                bond_lengths.append(d)

    avg_cn = float(np.nanmean(cn_list)) if cn_list else None
    mean_bond_length = float(np.mean(bond_lengths)) if bond_lengths else None
    bond_length_std = float(np.std(bond_lengths)) if bond_lengths else None

    # Composition-based features
    comp = structure.composition
    valence_electron_count = 0
    en_list = []

    for elem, amt in comp.get_el_amt_dict().items():
        e = Element(elem)
        try:
            full_es = e.full_electronic_structure
            max_n = max(t[0] for t in full_es)
            valence = sum(t[2] for t in full_es if t[0] == max_n)
        except Exception:
            valence = getattr(e, "valence", 0)

        valence = valence or 0
        valence_electron_count += int(valence) * int(amt)

        en = getattr(e, "X", None)
        if en is not None:
            en_list.extend([en] * int(amt))

    electronegativity_mean = float(np.mean(en_list)) if en_list else None
    electronegativity_difference = (
        float(max(en_list) - min(en_list)) if len(en_list) >= 2 else None
    )

    result = {
        "formula": comp.formula,
        "pymatgen_structure": structure.as_dict(),
        "structural_features": {
            "space_group_number": sg_number,
            "space_group_symbol": sg_symbol,
            "crystal_system": crystal_system,
            "volume_per_atom": volume_per_atom,
            "density": density,
            "valence_electron_count": valence_electron_count,
            "avg_coordination_number": avg_cn,
            "mean_bond_length": mean_bond_length,
            "bond_length_std": bond_length_std,
            "electronegativity_mean": electronegativity_mean,
            "electronegativity_difference": electronegativity_difference,
        },
    }

    return result
