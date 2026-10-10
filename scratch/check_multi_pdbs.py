import sys
from pathlib import Path
import pandas as pd
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.app.ml import load_all_splits

tr, va, te, _ = load_all_splits()
all_pos_pdbs = tr.pdb_ids + va.pdb_ids + te.pdb_ids
counts = Counter(all_pos_pdbs)
multi_pdbs = {k: v for k, v in counts.items() if v > 1}
print("PDBs with multiple positive samples:", multi_pdbs)
