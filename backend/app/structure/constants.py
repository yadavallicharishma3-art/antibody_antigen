"""Standard biochemical constants and amino-acid dictionaries for structural analysis."""

# 20 Standard Amino Acids Mapping
STANDARD_AA_3TO1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

STANDARD_AA_1TO3 = {v: k for k, v in STANDARD_AA_3TO1.items()}

STANDARD_AA_LETTERS = sorted(list(STANDARD_AA_3TO1.values()))
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(STANDARD_AA_LETTERS)}

# IMGT Standard CDR Residue Numbering Ranges
IMGT_CDR_RANGES = {
    "1": (27, 38),
    "2": (56, 65),
    "3": (105, 117),
}

# Standard Backbone Atom Names to filter when isolating side-chain heavy atoms later
BACKBONE_ATOM_NAMES = {"N", "CA", "C", "O", "OXT"}
