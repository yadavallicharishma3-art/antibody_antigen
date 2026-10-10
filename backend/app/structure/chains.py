"""Chain inventory and biological entity identification (Antibody Heavy/Light and Antigen)."""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

from .models import ComplexData, ChainData

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
METADATA_CSV_PATH = BASE_DIR / "data" / "abdb" / "abag_split.csv"

# Global lazy-loaded metadata cache
_METADATA_DF: Optional[pd.DataFrame] = None


def get_metadata_df() -> Optional[pd.DataFrame]:
    """Retrieve or load SAbDab metadata dataframe."""
    global _METADATA_DF
    if _METADATA_DF is None:
        if METADATA_CSV_PATH.exists():
            _METADATA_DF = pd.read_csv(METADATA_CSV_PATH)
    return _METADATA_DF


def parse_delimited_chains(raw_value: Any) -> List[str]:
    """Parse delimited chain strings (e.g. 'A/B', 'A,B', 'A;B', 'P') into clean list."""
    if pd.isna(raw_value) or not raw_value:
        return []
    parts = re.split(r"[,;/\s]+", str(raw_value).strip())
    return [p.strip() for p in parts if p.strip() and p.strip() not in {"/", "-", "null", "None"}]


def find_sabdab_record(pdb_id: str) -> Optional[pd.Series]:
    """Find matching SAbDab metadata record for a 4-letter PDB ID."""
    df = get_metadata_df()
    if df is None or df.empty:
        return None

    code = pdb_id.strip().upper()
    matches = df[df["PDB_ID"].str.upper().str.contains(code, na=False)]
    if not matches.empty:
        return matches.iloc[0]
    return None


def generate_chain_inventory(complex_data: ComplexData) -> Dict[str, Dict[str, Any]]:
    """Generate diagnostic inventory of all chains in a parsed structure.

    Returns:
        Dict mapping chain_id -> dict of counts (total residues, standard residues, atoms).
    """
    inventory = {}
    for chain_id, chain_data in complex_data.chains.items():
        total_res = len(chain_data.residues)
        std_res = len(chain_data.standard_residues)
        total_atoms = sum(len(r.atoms) for r in chain_data.residues)
        std_atoms = sum(len(r.atoms) for r in chain_data.standard_residues)

        inventory[chain_id] = {
            "total_residues": total_res,
            "standard_residues": std_res,
            "total_atoms": total_atoms,
            "standard_atoms": std_atoms,
            "sequence_length": len(chain_data.sequence),
        }
    return inventory


def identify_complex_entities(
    complex_data: ComplexData,
    sabdab_record: Optional[pd.Series] = None,
    allow_metadata_lookup: bool = True,
) -> ComplexData:
    """Identify antibody heavy/light chains and antigen chains with recorded evidence.

    Uses SAbDab metadata as primary source of truth.
    Never assumes H=Heavy, L=Light, or A=Antigen.

    Args:
        complex_data: Parsed structure to annotate.
        sabdab_record: Optional pre-fetched SAbDab metadata row.
        allow_metadata_lookup: If True and sabdab_record is None, attempts lookup by PDB ID.

    Returns:
        Modified ComplexData with annotated antibody and antigen chains.
    """
    if sabdab_record is not None:
        record = sabdab_record
    elif allow_metadata_lookup:
        record = find_sabdab_record(complex_data.pdb_id)
    else:
        record = None

    if record is not None:
        # 1. Identify Heavy chain
        h_chain = str(record["Hchain"]).strip() if not pd.isna(record["Hchain"]) else None
        if h_chain and h_chain in complex_data.chains:
            complex_data.antibody.heavy_chain_id = h_chain
            complex_data.chains[h_chain].chain_type = "HEAVY"
            complex_data.chains[h_chain].evidence = "SAbDab metadata (Hchain)"
        elif h_chain:
            complex_data.warnings.append(
                f"SAbDab designated heavy chain '{h_chain}' not found in structure models."
            )

        # 2. Identify Light chain
        l_chain = str(record["Lchain"]).strip() if not pd.isna(record["Lchain"]) else None
        if l_chain and l_chain in complex_data.chains:
            complex_data.antibody.light_chain_id = l_chain
            complex_data.chains[l_chain].chain_type = "LIGHT"
            complex_data.chains[l_chain].evidence = "SAbDab metadata (Lchain)"
        elif l_chain:
            complex_data.warnings.append(
                f"SAbDab designated light chain '{l_chain}' not found in structure models."
            )

        complex_data.antibody.evidence = "SAbDab database annotation"

        # 3. Identify Antigen chains (support multiple subunits)
        raw_ag = record.get("agchains", "")
        ag_chains = parse_delimited_chains(raw_ag)
        valid_ag = [cid for cid in ag_chains if cid in complex_data.chains]

        complex_data.antigen_chain_ids = valid_ag
        complex_data.antigen_evidence = f"SAbDab metadata (agchains={raw_ag})"
        for cid in valid_ag:
            complex_data.chains[cid].chain_type = "ANTIGEN"
            complex_data.chains[cid].evidence = f"SAbDab metadata (agchains={raw_ag})"

        if not valid_ag and ag_chains:
            complex_data.warnings.append(
                f"SAbDab designated antigen chains '{ag_chains}' not present in structure models."
            )

    else:
        # Explicit ambiguity reporting when no metadata is available
        complex_data.warnings.append(
            f"No SAbDab metadata found for {complex_data.pdb_id}. Cannot reliably assign antibody/antigen without annotation."
        )

    return complex_data
