import numpy as np
from pathlib import Path

base = Path('.')
ejo_20x21 = np.load(base / 'data' / 'processed' / 'representations' / 'pdb_00001ejo_H_L.npy')
print("ejo_20x21 shape:", ejo_20x21.shape)

# Also check canonical 20x20 matrices if saved
ejo_20x20_path = base / 'data' / 'processed' / 'matrices' / '1EJO_matrix_20x20.npy'
if ejo_20x20_path.exists():
    ejo_20x20 = np.load(ejo_20x20_path)
    print("ejo_20x20 shape:", ejo_20x20.shape)
    # in 20x20: lower triangle is antigen, upper triangle is antibody
    # let's check row 0
    print("20x21 row 0:", ejo_20x21[0])
    print("20x21 row 5:", ejo_20x21[5])
