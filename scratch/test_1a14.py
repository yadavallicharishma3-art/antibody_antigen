import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))

from backend.app.structure import (
    resolve_structure_path,
    analyze_biological_entities,
    build_interaction_representation,
)

pdb = "1A14"
print(f"Downloading/resolving {pdb}...")
p = resolve_structure_path(pdb, allow_download=True)
print(f"Resolved path: {p}")

comp = analyze_biological_entities(p)
print(f"Antibody: H={comp.antibody.heavy_chain_id}, L={comp.antibody.light_chain_id}")
print(f"Antigen chains: {comp.antigen_chain_ids}")
print(f"CDRs: {[k for k, v in comp.antibody.cdrs.items() if v.mapping_status == 'MATCH']}")

rep = build_interaction_representation(comp)
print(f"Inter contacts: {rep.intermolecular.contact_count}")
print(f"CDR interface residues: {rep.intermolecular.cdr_interface_count}")
print(f"Antigen interface residues: {rep.intermolecular.antigen_interface_count}")
print(f"Tensor shape: {rep.tensor_20x21.shape}")
print(f"Nonzero cells: {rep.stats_20x21.nonzero_count}")
print(f"SHA256: {rep.stats_20x21.sha256}")
