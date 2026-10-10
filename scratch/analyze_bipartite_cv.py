import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.app.ml import load_all_splits, generate_mismatched_negatives

tr, va, te, _ = load_all_splits()
tr_b = generate_mismatched_negatives(tr, seed=42)
va_b = generate_mismatched_negatives(va, seed=42)
te_b = generate_mismatched_negatives(te, seed=42)

all_X = np.concatenate([tr_b.X, va_b.X, te_b.X], axis=0)
all_y = np.concatenate([tr_b.y, va_b.y, te_b.y], axis=0)
all_samples = tr_b.sample_ids + va_b.sample_ids + te_b.sample_ids
all_pdbs = tr_b.pdb_ids + va_b.pdb_ids + te_b.pdb_ids
all_meta = tr_b.metadata + va_b.metadata + te_b.metadata

# Check StratifiedKFold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
skf_folds = list(skf.split(all_X, all_y))

print("=== StratifiedKFold Overlap Audit ===")
for i, (train_idx, val_idx) in enumerate(skf_folds):
    tr_samples = set([all_samples[k] for k in train_idx])
    val_samples = set([all_samples[k] for k in val_idx])
    sample_overlap = tr_samples & val_samples
    assert len(sample_overlap) == 0, f"Sample overlap in fold {i+1}!"

    # Positive PDB overlap
    tr_pos_pdbs = set([all_pdbs[k] for k in train_idx if all_y[k] == 1])
    val_pos_pdbs = set([all_pdbs[k] for k in val_idx if all_y[k] == 1])
    pos_overlap = tr_pos_pdbs & val_pos_pdbs
    print(f"Fold {i+1:2d}: Val samples={len(val_idx)} (Pos={np.sum(all_y[val_idx]==1)}, Neg={np.sum(all_y[val_idx]==0)}) | Pos PDB overlap={pos_overlap}")

# Let's inspect GroupKFold by PDB:
# Can negative samples have a single PDB? A negative sample is a hybrid of two PDBs (e.g. 1EJO and 8PWH).
# If a negative sample has ab from 1EJO and ag from 8PWH, if 1EJO is in fold 1 and 8PWH is in fold 2, where does 1EJO_8PWH go?
# Any group assignment for a pair of entities can only group by one entity (either Ab or Ag), but not both!
# This is a classic dyadic/bipartite graph problem in pairwise bioinformatics.
