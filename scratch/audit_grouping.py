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

# Let's inspect the groups:
# What groups exist for each sample?
# For positive samples: pdb_id is e.g. 1EJO
# For negative samples: pdb_id is e.g. 1EJO_8PWH, which combines two PDBs!
print("Unique positive PDBs:", len(set([all_pdbs[i] for i in range(len(all_pdbs)) if all_y[i] == 1])))
print("Unique negative PDBs:", len(set([all_pdbs[i] for i in range(len(all_pdbs)) if all_y[i] == 0])))

# In negative samples, the sample is formed by antibody from sample A and antigen from sample B.
# If a negative sample is grouped, how would groups be assigned?
# If we try StratifiedGroupKFold using primary PDB:
# Can a sample have two PDBs? Negative samples have ab_source and ag_source from different PDBs!
# Let's check how many negative samples have ab_source and ag_source from different PDBs:
diff_pdb_negs = 0
for i, m in enumerate(all_meta):
    if all_y[i] == 0:
        ab_src = m.get('ab_source', '')
        ag_src = m.get('ag_source', '')
        if ab_src != ag_src:
            diff_pdb_negs += 1
print(f"Negative samples with distinct ab and ag sources: {diff_pdb_negs} / {np.sum(all_y == 0)}")

# Check how StratifiedGroupKFold would behave if we used ab_source or pdb_id
try:
    # Use ab_source as group
    groups = [m.get('ab_source', all_samples[i]) for i, m in enumerate(all_meta)]
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    sgkf_folds = list(sgkf.split(all_X, all_y, groups=groups))
    print("\nStratifiedGroupKFold with ab_source group:")
    for idx, (tr, va) in enumerate(sgkf_folds):
        y_va = all_y[va]
        print(f"Fold {idx+1:2d}: Val={len(va)} (Pos={np.sum(y_va==1)}, Neg={np.sum(y_va==0)})")
except Exception as e:
    print("StratifiedGroupKFold failed:", e)
