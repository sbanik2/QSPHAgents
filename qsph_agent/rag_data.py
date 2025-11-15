# rag_data.py
from __future__ import annotations

import json
import os
import random
from typing import Dict, Tuple

import numpy as np

from .database_genrator import generate_database  # your existing module


# ---------- DB generation / loading ----------

def get_datafile_from_config(config: dict | None = None) -> str:
    cfg = config or {}
    db_cfg = cfg.get("database", {})
    return db_cfg.get("output_file", "materials_database.json")


def build_database_if_needed(mp_api_key: str, config: dict | None = None) -> str:
    """
    Generate the materials database JSON if it does not exist yet.
    Returns the path to the data file.
    """
    cfg = config or {}
    db_cfg = cfg.get("database", {})
    species = db_cfg.get("species", ["Si"])
    cutoff = db_cfg.get("cutoff", 5.0)

    datafile = get_datafile_from_config(cfg)

    if os.path.exists(datafile):
        print(f"[Info] Database file '{datafile}' already exists – skipping generation.")
        return datafile

    print(f"[Info] Generating database file '{datafile}' for species {species}...")
    generate_database(
        mp_api_key,
        species=species,
        cutoff=cutoff,
        output_file=datafile,
        config=cfg,
    )
    return datafile


def load_and_split_database(
    datafile: str,
    train_fraction: float = 0.8,
    seed: int = 786,
) -> Tuple[Dict, Dict, Dict]:
    """
    Load the JSON database and split into train/test dictionaries.
    Returns (train_set, test_set, full_data).
    """
    if not os.path.exists(datafile):
        raise FileNotFoundError(f"Database file '{datafile}' not found.")

    with open(datafile, "r") as infile:
        data = json.load(infile)

    random.seed(seed)
    keys = list(data.keys())
    random.shuffle(keys)

    split_idx = int(train_fraction * len(keys))
    train_set = {k: data[k] for k in keys[:split_idx]}
    test_set = {k: data[k] for k in keys[split_idx:]}

    print(f"Train size: {len(train_set)}, Test size: {len(test_set)}")
    return train_set, test_set, data


# ---------- RAG feature utilities ----------

def structural_vector(struct_features: dict) -> np.ndarray:
    numeric_keys = [
        "space_group_number",
        "volume_per_atom",
        "density",
        "valence_electron_count",
        "avg_coordination_number",
        "mean_bond_length",
        "bond_length_std",
        "electronegativity_mean",
        "electronegativity_difference",
    ]
    return np.array(
        [struct_features.get(k, 0.0) or 0.0 for k in numeric_keys],
        dtype=float,
    )


def retrieve_similar_structures(test_entry: dict, train_data: dict, top_k: int = 5):
    test_vec = structural_vector(test_entry["structure"]["structural_features"])
    distances = []
    for uid, entry in train_data.items():
        train_vec = structural_vector(entry["structure"]["structural_features"])
        dist = np.linalg.norm(test_vec - train_vec)
        distances.append((uid, dist))
    distances.sort(key=lambda x: x[1])
    return [(uid, train_data[uid]) for uid, _ in distances[:top_k]]


def generate_text_summary(structure_entry: dict) -> str:
    """
    Generate a concise human-readable text summary from structural features.
    """
    features = structure_entry["structural_features"]
    formula = structure_entry.get("formula", "Unknown")
    sg_symbol = features.get("space_group_symbol", "Unknown")
    crystal_system = features.get("crystal_system", "Unknown")
    crystal_system = crystal_system.capitalize() if isinstance(crystal_system, str) else str(crystal_system)
    avg_cn = features.get("avg_coordination_number")
    mean_bond_length = features.get("mean_bond_length")

    summary_parts = [f"{crystal_system} {formula} ({sg_symbol})"]

    if avg_cn is not None:
        summary_parts.append(f"avg coordination ≈ {avg_cn:.1f}")
    if mean_bond_length is not None:
        summary_parts.append(f"mean bond length ≈ {mean_bond_length:.2f} Å")

    return ", ".join(summary_parts) + "."


def build_context_from_neighbors(test_entry: dict, neighbors: list) -> str:
    """
    Build a text context block from nearest-neighbor structures and their DOS descriptions.
    """
    lines = []
    lines.append("Top structurally similar materials:")

    for uid, entry in neighbors:
        neighbor_summary = generate_text_summary(entry["structure"])
        lines.append(f"- {entry['structure']['formula']}: {neighbor_summary}")
        if "dos" in entry and "dos_description" in entry["dos"]:
            lines.append(f"  DOS: {entry['dos']['dos_description']}")

    return "\n".join(lines)
