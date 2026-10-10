import pandas as pd
from pathlib import Path

base = Path('.')
rej_df = pd.read_csv(base / 'data' / 'processed' / 'rejected_samples.csv')
for pdb in ['1VPO', '1C5C', '1A3L', '1AJ7', '1BBD', '1CF8', '1D5I', '1D6V']:
    sub = rej_df[rej_df['pdb_id'].str.upper() == pdb]
    if not sub.empty:
        for _, r in sub.iterrows():
            print(f"{pdb}: reason={r['reason']}, stage={r['stage_failed']}, error={r['detailed_error']}")
    else:
        print(f"{pdb}: not in rejected_samples")
