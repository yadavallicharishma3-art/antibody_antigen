"""Structural processing, chain identification, CDR detection, contact extraction, and matrix representations."""

from typing import Union, Optional, Any
from pathlib import Path

from .models import AtomData, ResidueData, ChainData, CDRData, AntibodyData, ComplexData
from .constants import STANDARD_AA_3TO1, STANDARD_AA_1TO3, STANDARD_AA_LETTERS, IMGT_CDR_RANGES, BACKBONE_ATOM_NAMES
from .loader import resolve_structure_path, normalize_pdb_id, StructureLoadError
from .parser import parse_structure_file, load_and_parse_complex, StructureParseError
from .chains import generate_chain_inventory, identify_complex_entities, find_sabdab_record
from .cdr import identify_antibody_cdrs
from .contacts import (
    get_sidechain_heavy_atoms,
    find_intermolecular_contacts,
    find_intramolecular_contacts,
    ContactRecord,
    IntermolecularContactResult,
    IntramolecularContactResult,
)
from .representation import (
    AMINO_3_CODES,
    AMINO_1_CODES,
    AA_INDEX_MAP,
    MatrixStatistics,
    RepresentationResult,
    build_upper_triangle,
    compute_matrix_statistics,
    build_interaction_representation,
    export_representation,
)


def analyze_biological_entities(
    identifier_or_path: Union[str, Path],
    sabdab_record: Optional[Any] = None,
) -> ComplexData:
    """End-to-end Phase 2 entrypoint: loads structure, parses chains, identifies antibody/antigen,
    and maps CDR residues to structural coordinates.

    Args:
        identifier_or_path: PDB code (e.g. '1EJO') or path to .cif/.mmcif/.pdb.
        sabdab_record: Optional pre-fetched SAbDab metadata record for this complex/instance.

    Returns:
        Fully annotated ComplexData with chains, antibody heavy/light, antigen, and CDRs.
    """
    complex_data = load_and_parse_complex(identifier_or_path)
    complex_data = identify_complex_entities(complex_data, sabdab_record=sabdab_record)
    complex_data = identify_antibody_cdrs(complex_data, sabdab_record=sabdab_record)
    return complex_data


__all__ = [
    "AtomData",
    "ResidueData",
    "ChainData",
    "CDRData",
    "AntibodyData",
    "ComplexData",
    "STANDARD_AA_3TO1",
    "STANDARD_AA_1TO3",
    "STANDARD_AA_LETTERS",
    "IMGT_CDR_RANGES",
    "BACKBONE_ATOM_NAMES",
    "resolve_structure_path",
    "normalize_pdb_id",
    "StructureLoadError",
    "parse_structure_file",
    "load_and_parse_complex",
    "StructureParseError",
    "generate_chain_inventory",
    "identify_complex_entities",
    "find_sabdab_record",
    "identify_antibody_cdrs",
    "analyze_biological_entities",
    "get_sidechain_heavy_atoms",
    "find_intermolecular_contacts",
    "find_intramolecular_contacts",
    "ContactRecord",
    "IntermolecularContactResult",
    "IntramolecularContactResult",
    "AMINO_3_CODES",
    "AMINO_1_CODES",
    "AA_INDEX_MAP",
    "MatrixStatistics",
    "RepresentationResult",
    "build_upper_triangle",
    "compute_matrix_statistics",
    "build_interaction_representation",
    "export_representation",
]

