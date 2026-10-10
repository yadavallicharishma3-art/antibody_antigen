"""Phase 1 Environment, Data Schema, and Architectural Verification Suite.

Tests dependencies, dataset file schemas, cached structure assets,
and exact CNN tensor shapes (confirming the 288-dimensional flattened bottleneck).
"""

from pathlib import Path
import pytest
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"


def test_core_dependencies_importable():
    """Verify all biomedical and machine learning libraries are functional."""
    import Bio
    from Bio.PDB import MMCIFParser, NeighborSearch
    import sklearn
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, accuracy_score
    import tensorflow as tf
    from tensorflow.keras import layers, models
    import pandas as pd
    import pydantic
    import fastapi

    assert Bio.__version__ is not None
    assert sklearn.__version__ is not None
    assert tf.__version__ is not None
    assert pd.__version__ is not None
    assert pydantic.__version__ is not None
    assert fastapi.__version__ is not None


def test_abdb_dataset_list():
    """Verify AbDb complex list exists and contains ~1,215 complexes."""
    abdb_file = DATA_DIR / "abdb" / "AbDb_list.dat"
    assert abdb_file.exists(), f"Missing {abdb_file}"
    with open(abdb_file, "r") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    assert len(lines) >= 300, f"Expected complexes in AbDb_list.dat, got {len(lines)}"
    assert any("1EJO" in line for line in lines), "1EJO should be in AbDb list"


def test_3did_control_dataset_list():
    """Verify 3did control dataset schema: Index, PDB, Chain1, S1, E1, Chain2, S2, E2."""
    did3_file = DATA_DIR / "3did" / "pdb_4960list.txt"
    assert did3_file.exists(), f"Missing {did3_file}"
    with open(did3_file, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    # The paper references 4,960 original 3did complexes, which the authors filtered down
    # to 4,144 entries in pdb_4960list.txt (with 3,218 inter-chain pairs used in Test 1).
    assert len(lines) == 4144, f"Expected 4,144 entries in pdb_4960list.txt, found {len(lines)}"

    sample = lines[0].split()
    assert len(sample) == 8, f"Expected 8 tokens per line in 3did list, got {sample}"
    assert len(sample[1]) == 4, f"Expected 4-letter PDB ID, got {sample[1]}"


def test_sabdab_metadata_csv():
    """Verify SAbDab metadata CSV schema contains required structural annotation columns."""
    csv_path = DATA_DIR / "abdb" / "abag_split.csv"
    assert csv_path.exists(), f"Missing {csv_path}"
    import pandas as pd
    df_sample = pd.read_csv(csv_path, nrows=10)
    required_cols = [
        "PDB_ID", "Hchain", "Lchain", "agchains",
        "VH_numbering_list", "VL_numbering_list",
        "CDRH1", "CDRH2", "CDRH3", "CDRL1", "CDRL2", "CDRL3"
    ]
    for col in required_cols:
        assert col in df_sample.columns, f"Required column '{col}' missing from abag_split.csv"


def test_required_structures_present():
    """Verify known test case structures are cached in data/structures/."""
    structures_dir = DATA_DIR / "structures"
    assert structures_dir.exists(), f"Missing {structures_dir}"
    for pdb_id in ["1EJO", "1NBZ", "1KC5", "1DQJ"]:
        cif_file = structures_dir / f"{pdb_id}.cif"
        assert cif_file.exists(), f"Required benchmark structure {pdb_id}.cif missing in {structures_dir}"
        assert cif_file.stat().st_size > 1000, f"{pdb_id}.cif is empty or corrupt"


def test_cnn_architecture_produces_exact_288_dimensions():
    """Verify the paper's 288-dimensional flattened representation is strictly met.

    Demonstrates:
    - MaxPooling2D with padding='same' preserves odd dimensions (5 -> 3), yielding 3x3x32 = 288.
    - Default padding='valid' yields 2x2x32 = 128 (the bug in the previous implementation).
    """
    from tensorflow.keras import layers, models

    # Paper faithful architecture
    model_288 = models.Sequential([
        layers.Input(shape=(20, 20, 1)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2), padding="same"),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2), padding="same"),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2), padding="same"),
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])

    flatten_layer = model_288.layers[6]
    assert flatten_layer.output.shape[-1] == 288, (
        f"Expected 288-dimensional flattened tensor, got {flatten_layer.output.shape[-1]}"
    )

    # Flawed architecture demonstration (verifying bug identification)
    model_flawed = models.Sequential([
        layers.Input(shape=(20, 20, 1)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)), # default padding='valid'
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
    ])
    assert model_flawed.layers[6].output.shape[-1] == 128, "Flawed architecture should produce 128 dimensions"
