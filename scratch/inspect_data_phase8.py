import sys
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.app.ml import load_all_splits, generate_mismatched_negatives

train_bundle, val_bundle, test_bundle, leakage = load_all_splits()
print(f"RAW Phase 4 Splits:")
print(f"Train: len={len(train_bundle)}, pos={train_bundle.positive_count}, neg={train_bundle.negative_count}, unique_pdbs={len(train_bundle.unique_pdbs)}")
print(f"Val:   len={len(val_bundle)}, pos={val_bundle.positive_count}, neg={val_bundle.negative_count}, unique_pdbs={len(val_bundle.unique_pdbs)}")
print(f"Test:  len={len(test_bundle)}, pos={test_bundle.positive_count}, neg={test_bundle.negative_count}, unique_pdbs={len(test_bundle.unique_pdbs)}")

all_pdbs = train_bundle.unique_pdbs | val_bundle.unique_pdbs | test_bundle.unique_pdbs
print(f"Total positive samples across splits: {len(train_bundle) + len(val_bundle) + len(test_bundle)}")
print(f"Total unique PDBs across splits: {len(all_pdbs)}")

train_b = generate_mismatched_negatives(train_bundle, seed=42)
val_b = generate_mismatched_negatives(val_bundle, seed=42)
test_b = generate_mismatched_negatives(test_bundle, seed=42)
print(f"\nAfter generate_mismatched_negatives:")
print(f"Train: len={len(train_b)}, pos={train_b.positive_count}, neg={train_b.negative_count}")
print(f"Val:   len={len(val_b)}, pos={val_b.positive_count}, neg={val_b.negative_count}")
print(f"Test:  len={len(test_b)}, pos={test_b.positive_count}, neg={test_b.negative_count}")
total_samples = len(train_b) + len(val_b) + len(test_b)
print(f"Total: {total_samples} samples (pos: {train_b.positive_count + val_b.positive_count + test_b.positive_count}, neg: {train_b.negative_count + val_b.negative_count + test_b.negative_count})")
