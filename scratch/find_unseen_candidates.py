import pandas as pd
from pathlib import Path

base = Path('.')
df = pd.read_csv(base / 'data' / 'abdb' / 'abag_split.csv')

train_df = pd.read_csv(base / 'data' / 'processed' / 'train.csv')
val_df = pd.read_csv(base / 'data' / 'processed' / 'validation.csv')
test_df = pd.read_csv(base / 'data' / 'processed' / 'test.csv')

used_pdbs = set(train_df['pdb_id'].str.upper()) | set(val_df['pdb_id'].str.upper()) | set(test_df['pdb_id'].str.upper())

# Look for test split structures with valid protein antigens
test_candidates = df[
    (df['ab_ag_split'] == 'test') &
    (df['agtypes'].str.contains('PROTEIN|PEPTIDE', na=False, case=False)) &
    (df['Hchain'].notna()) &
    (df['Lchain'].notna()) &
    (df['agchains'].notna())
]

clean_cands = []
for _, row in test_candidates.iterrows():
    raw_pdb = str(row['PDB_ID'])
    clean = raw_pdb.replace('pdb_0000', '').upper() if 'pdb_0000' in raw_pdb else raw_pdb.upper()
    if clean not in used_pdbs:
        clean_cands.append((clean, row['INSTANCE'], row['Hchain'], row['Lchain'], row['agchains'], row['agtypes'], row['ab_ag_cluster']))

print(f"Total test candidate complexes not used in any split: {len(clean_cands)}")
for c in clean_cands[:15]:
    print(c)
