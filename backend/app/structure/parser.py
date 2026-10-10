"""Macromolecular structure parser using Biopython for PDB and mmCIF formats."""

from pathlib import Path
from typing import Union, Dict, List, Optional
import numpy as np

from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Structure import Structure

from .constants import STANDARD_AA_3TO1, BACKBONE_ATOM_NAMES
from .models import AtomData, ResidueData, ChainData, ComplexData
from .loader import resolve_structure_path, normalize_pdb_id, StructureLoadError


class StructureParseError(Exception):
    """Raised when parsing macromolecular coordinates fails."""
    pass


def parse_structure_file(file_path: Union[str, Path], pdb_id: Optional[str] = None) -> ComplexData:
    """Parse a PDB or mmCIF structure file into standardized structural models.

    Args:
        file_path: Path to .cif, .mmcif, or .pdb file.
        pdb_id: Optional PDB ID. If None, inferred from filename.

    Returns:
        ComplexData containing parsed chains, standard residues, and atomic coordinates.
    """
    path = Path(file_path)
    if not path.exists():
        raise StructureParseError(f"File not found: {path}")

    code = pdb_id.upper() if pdb_id else normalize_pdb_id(path.stem)
    ext = path.suffix.lower()

    if ext in [".cif", ".mmcif"]:
        parser = MMCIFParser(QUIET=True)
    elif ext in [".pdb", ".ent"]:
        parser = PDBParser(QUIET=True)
    else:
        # Default to mmCIF parser if unrecognized extension
        parser = MMCIFParser(QUIET=True)

    try:
        biopython_struct: Structure = parser.get_structure(code, str(path))
    except Exception as exc:
        raise StructureParseError(f"Biopython failed to parse '{path}': {exc}") from exc

    models = list(biopython_struct.get_models())
    if not models:
        raise StructureParseError(f"No models found in structure '{path}'")

    # The paper evaluates the primary model (model 0)
    primary_model = models[0]
    parsed_chains: Dict[str, ChainData] = {}

    for chain in primary_model:
        chain_id = str(chain.id).strip()
        residues_list: List[ResidueData] = []

        for residue in chain.get_residues():
            hetflag, resseq, inscode = residue.id
            resname = str(residue.get_resname()).strip().upper()
            is_standard = (hetflag == " ") and (resname in STANDARD_AA_3TO1)
            one_letter = STANDARD_AA_3TO1.get(resname, "") if is_standard else ""

            # Extract atoms
            atoms_list: List[AtomData] = []
            for atom in residue.get_atoms():
                name = str(atom.get_name()).strip().upper()
                element = (getattr(atom, "element", "") or "").strip().upper()
                is_heavy = (element != "H") and (not name.startswith("H"))
                is_sidechain = name not in BACKBONE_ATOM_NAMES

                atoms_list.append(AtomData(
                    name=name,
                    element=element,
                    coord=np.array(atom.coord, dtype=np.float32),
                    is_sidechain=is_sidechain,
                    is_heavy=is_heavy,
                ))

            residues_list.append(ResidueData(
                chain_id=chain_id,
                residue_number=int(resseq),
                insertion_code=str(inscode),
                residue_name=resname,
                one_letter_code=one_letter,
                is_standard=is_standard,
                atoms=atoms_list,
            ))

        parsed_chains[chain_id] = ChainData(
            chain_id=chain_id,
            residues=residues_list,
        )

    return ComplexData(
        pdb_id=code,
        source_path=str(path),
        models_count=len(models),
        chains=parsed_chains,
    )


def load_and_parse_complex(identifier_or_path: Union[str, Path]) -> ComplexData:
    """Convenience helper to resolve and parse a structure in one call."""
    path = resolve_structure_path(identifier_or_path)
    return parse_structure_file(path)
