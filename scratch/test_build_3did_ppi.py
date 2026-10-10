import sys
from pathlib import Path
import numpy as np
import pandas as pd
from Bio.PDB.NeighborSearch import NeighborSearch
from scipy.spatial import cKDTree

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.app.structure.parser import parse_structure_file
from backend.app.structure.loader import resolve_structure_path
from backend.app.structure.contacts import find_intramolecular_contacts, get_sidechain_heavy_atoms
from backend.app.structure.representation import build_upper_triangle

# Load SAbDab PDBs to ensure strict exclusion of antibody complexes from negative class
abag_df = pd.read_csv(BASE_DIR / "data" / "abdb" / "abag_split.csv")
antibody_pdbs = set(abag_df['PDB_ID'].dropna().str.replace('pdb_0000', '').str.upper())

did_file = BASE_DIR / "data" / "3did" / "pdb_4960list.txt"
candidates = []
with open(did_file) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 8 and parts[2] != parts[5]:
            pdb = parts[1].upper()
            if pdb not in antibody_pdbs:
                candidates.append((pdb, parts[2], parts[5]))

print(f"Total non-antibody inter-chain candidates in 3did: {len(candidates)}")

# Process until 37 valid accepted negative samples
accepted_ppi = []
rejected_ppi = []
seen_pdbs = set()

for idx, (pdb, c1, c2) in enumerate(candidates):
    if len(accepted_ppi) >= 37:
        break
    if pdb in seen_pdbs:
        continue

    try:
        cif_path = resolve_structure_path(pdb, allow_download=True)
        complex_data = parse_structure_file(cif_path, pdb_id=pdb)
        if c1 not in complex_data.chains or c2 not in complex_data.chains:
            rejected_ppi.append({"pdb": pdb, "reason": f"Chains {c1}/{c2} not found in model 0"})
            continue

        chain_a = complex_data.chains[c1]
        chain_b = complex_data.chains[c2]

        atoms_a = [a for res in chain_a.residues for a in get_sidechain_heavy_atoms(res)]
        atoms_b = [a for res in chain_b.residues for a in get_sidechain_heavy_atoms(res)]
        if not atoms_a or not atoms_b:
            rejected_ppi.append({"pdb": pdb, "reason": "Missing sidechain heavy atoms"})
            continue

        coords_a = np.array([a.coord for a in atoms_a])
        coords_b = np.array([a.coord for a in atoms_b])
        tree_b = cKDTree(coords_b)
        tree_a = cKDTree(coords_a)

        p1_interface = [res for res in chain_a.residues if (ra := get_sidechain_heavy_atoms(res)) and any(len(p) > 0 for p in tree_b.query_ball_point(np.array([a.coord for a in ra]), r=5.0))]
        p2_interface = [res for res in chain_b.residues if (ra := get_sidechain_heavy_atoms(res)) and any(len(p) > 0 for p in tree_a.query_ball_point(np.array([a.coord for a in ra]), r=5.0))]

        if not p1_interface or not p2_interface:
            rejected_ppi.append({"pdb": pdb, "reason": "No intermolecular interface residues at 5A"})
            continue

        p1_intra = find_intramolecular_contacts(p1_interface, cutoff=5.0)
        p2_intra = find_intramolecular_contacts(p2_interface, cutoff=5.0)

        if len(p1_intra.contacts) == 0 or len(p2_intra.contacts) == 0:
            rejected_ppi.append({"pdb": pdb, "reason": "Zero intramolecular contacts in interface"})
            continue

        raw_p1, norm_p1 = build_upper_triangle(p1_intra.pair_frequencies)
        raw_p2, norm_p2 = build_upper_triangle(p2_intra.pair_frequencies)

        if np.isnan(norm_p1).all() or np.isnan(norm_p2).all():
            rejected_ppi.append({"pdb": pdb, "reason": "Normalization produced all NaNs"})
            continue

        flipped_p1 = np.flip(norm_p1)
        cat_chart = np.concatenate([flipped_p1, norm_p2], axis=1)
        tensor_20x21 = np.array([row[~np.isnan(row)] for row in cat_chart], dtype=np.float32)

        if tensor_20x21.shape != (20, 21):
            rejected_ppi.append({"pdb": pdb, "reason": f"Invalid tensor shape: {tensor_20x21.shape}"})
            continue

        if np.isnan(tensor_20x21).any() or np.isinf(tensor_20x21).any():
            rejected_ppi.append({"pdb": pdb, "reason": "Tensor contains NaNs or Infs"})
            continue

        seen_pdbs.add(pdb)
        accepted_ppi.append({
            "sample_id": f"ppi_{pdb}_{c1}_{c2}",
            "pdb_id": pdb,
            "chain_1": c1,
            "chain_2": c2,
            "tensor_shape": tensor_20x21.shape,
            "nonzero_count": int(np.count_nonzero(tensor_20x21)),
            "p1_interface_count": len(p1_interface),
            "p2_interface_count": len(p2_interface),
            "p1_intra_contacts": len(p1_intra.contacts),
            "p2_intra_contacts": len(p2_intra.contacts),
        })
        print(f"Accepted [{len(accepted_ppi)}/37]: {pdb} ({c1}-{c2}) Nonzero={np.count_nonzero(tensor_20x21)}")

    except Exception as exc:
        rejected_ppi.append({"pdb": pdb, "reason": str(exc)})

print(f"\nFinished: Accepted={len(accepted_ppi)}, Rejected={len(rejected_ppi)}")
