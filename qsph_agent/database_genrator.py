import uuid
import json
from io import StringIO

from pymatgen.io.cif import CifParser
from mp_api.client import MPRester

# Relative imports (work inside a package)
from .mp_fetch import collect_materials_by_species
from .feature_extraction import extract_material_features
from .dos_description import get_dos_features


def generate_unique_id(prefix: str = "mat") -> str:
    """
    Generate a unique ID string with an optional prefix.

    Args:
        prefix (str): Prefix for the unique ID.

    Returns:
        str: A unique identifier.
    """
    return f"{prefix}_{uuid.uuid4().hex}"


def generate_database(
    MP_API_KEY: str,
    species: list,
    cutoff: float | None = None,
    output_file: str | None = None,
    config: dict | None = None,
):
    """
    Build a materials database by collecting structures and DOS features
    from the Materials Project.

    Args:
        MP_API_KEY (str): Materials Project API key.
        species (list): List of element symbols (e.g., ["Mo", "S"]).
        cutoff (float | None): Cutoff radius for structural feature extraction.
                               If None, uses config["database"]["cutoff"] or 5.0.
        output_file (str | None): Output filepath for JSON database.
                                  If None, uses config["database"]["output_file"]
                                  or "materials_database.json".
        config (dict | None): Global configuration dictionary.
                              Expected structure (example):

                                  {
                                      "database": {
                                          "cutoff": 5.0,
                                          "output_file": "materials_database.json",
                                          "primitive": true
                                      },
                                      "dos_features": {
                                          ...  # passed through to DOS routines
                                      }
                                  }

    Returns:
        None
    """
    cfg = config or {}
    db_cfg = cfg.get("database", {})

    # --- database-level parameters ---
    if cutoff is None:
        cutoff = db_cfg.get("cutoff", 5.0)

    if output_file is None:
        output_file = db_cfg.get("output_file", "materials_database.json")

    primitive_flag = db_cfg.get("primitive", True)

    # Fetch raw MP data
    with MPRester(MP_API_KEY) as mpr:
        raw_results = collect_materials_by_species(mpr, species=species)
        # keep only entries that actually have DOS data
        filtered_results = [r for r in raw_results if r.get("dos") is not None]

    if not filtered_results:
        print(
            f"No materials with DOS found for species "
            f"{', '.join(species)}. Nothing to write."
        )
        return

    print(
        f"Found {len(filtered_results)} materials containing "
        f"{', '.join(species)} with available DOS"
    )

    overall_data = {}

    # Process each result
    for i, result in enumerate(filtered_results):
        try:
            uid = generate_unique_id()

            # ------------ DOS FEATURES ------------ #
            # Pass full material_doc + global config into DOS pipeline
            dos_info, desc_dict, description = get_dos_features(
                material_doc=result,
                config=cfg,
            )

            # If DOS processing failed, skip this entry
            if dos_info is None:
                print(f"[Warning] Skipping material {i} due to DOS error.")
                continue

            # ------------ STRUCTURE FEATURES ------------ #
            cif_text = result.get("cif")
            if not cif_text:
                print(f"[Warning] Skipping material {i} due to missing CIF.")
                continue

            parser = CifParser(StringIO(cif_text))
            structure_info = extract_material_features(
                parser,
                cutoff=cutoff,
                primitive=primitive_flag,
            )

            # ------------ SAVE COMBINED DATA ------------ #
            overall_data[uid] = {
                "structure": structure_info,
                "dos": {
                    "dos_features": dos_info,
                    "dos_description": description,
                    "dos_description_dict": desc_dict,
                },
            }

        except Exception as e:
            print(f"[Warning] Skipping material {i} due to error: {e}")

    # ------------ SAVE JSON FILE ------------ #
    with open(output_file, "w") as f:
        json.dump(overall_data, f, indent=2)

    print(f"\nSaved database of {len(overall_data)} materials to {output_file}")
