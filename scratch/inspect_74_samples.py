import sys
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.app.ml import load_all_splits, generate_mismatched_negatives

train_bundle, val_bundle, test_bundle, leakage = load_all_splits()
train_b = generate_mismatched_negatives(train_bundle, seed=42)
val_b = generate_mismatched_negatives(val_bundle, seed=42)
test_b = generate_mismatched_negatives(test_bundle, seed=42)

# Check all 74 samples
all_X = np.concatenate([train_b.X, val_b.X, test_b.X], axis=0)
all_y = np.concatenate([train_b.y, val_b.y, test_b.y], axis=0)
all_samples = train_b.sample_ids + val_b.sample_ids + test_b.sample_ids
all_pdbs = train_b.pdb_ids + val_b.pdb_ids + test_b.pdb_ids
all_meta = train_b.metadata + val_b.metadata + test_b.metadata

print(f"Total samples: {len(all_samples)}")
print(f"Positive count: {np.sum(all_y == 1)}, Negative count: {np.sum(all_y == 0)}")
print(f"Unique sample IDs: {len(set(all_samples))}")

df = pd.DataFrame({
    'sample_id': all_samples,
    'pdb_id': all_pdbs,
    'label': all_y,
    'is_mismatched': [m.get('is_mismatched', False) for m in all_meta],
    'split_origin': ['train']*len(train_b) + ['val']*len(val_b) + ['test']*len(test_b),
    'ab_source': [m.get('ab_source', m.get('sample_id')) for m in all_meta],
    'ag_source': [m.get('ag_source', m.get('sample_id')) for m in all_meta],
})

print("\nSample preview:")
print(df.head(10))
print("\nUnique PDBs in positive samples:")
pos_pdbs = set(df[df['label'] == 1]['pdb_id'])
print(f"Count: {len(pos_pdbs)}")
print(sorted(list(pos_pdbs)))

print("\nNegative samples PDBs:")
neg_pdbs = set(df[df['label'] == 0]['pdb_id'])
print(f"Count: {len(neg_pdbs)}")

print("\nCheck if any sample_ids are duplicated:")
dups = df[df.duplicated(subset=['sample_id'])]
print(f"Duplicates: {len(dups)}")
