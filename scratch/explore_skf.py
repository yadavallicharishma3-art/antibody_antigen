import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.app.ml import load_all_splits, generate_mismatched_negatives

train_bundle, val_bundle, test_bundle, _ = load_all_splits()
train_b = generate_mismatched_negatives(train_bundle, seed=42)
val_b = generate_mismatched_negatives(val_bundle, seed=42)
test_b = generate_mismatched_negatives(test_bundle, seed=42)

all_X = np.concatenate([train_b.X, val_b.X, test_b.X], axis=0)
all_y = np.concatenate([train_b.y, val_b.y, test_b.y], axis=0)
all_samples = train_b.sample_ids + val_b.sample_ids + test_b.sample_ids
all_pdbs = train_b.pdb_ids + val_b.pdb_ids + test_b.pdb_ids
all_meta = train_b.metadata + val_b.metadata + test_b.metadata

print(f"Total samples: {len(all_y)}")

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
folds = list(skf.split(all_X, all_y))

print("StratifiedKFold (n_splits=10, shuffle=True, random_state=42):")
for fold_idx, (train_idx, val_idx) in enumerate(folds):
    y_tr, y_val = all_y[train_idx], all_y[val_idx]
    pos_tr, neg_tr = np.sum(y_tr == 1), np.sum(y_tr == 0)
    pos_val, neg_val = np.sum(y_val == 1), np.sum(y_val == 0)
    print(f"Fold {fold_idx + 1:2d}: Train={len(train_idx)} (Pos={pos_tr}, Neg={neg_tr}) | Val={len(val_idx)} (Pos={pos_val}, Neg={neg_val})")

# Let's inspect complex/source structure sharing in negative samples
# Each sample has ab_source and ag_source
print("\nInspecting PDB / complex sources:")
for i, m in enumerate(all_meta[:5]):
    print(all_samples[i], all_pdbs[i], m.get('ab_source'), m.get('ag_source'))
