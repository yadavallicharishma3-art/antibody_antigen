"""Contact extraction module for antibody-antigen complexes.

Implements:
1. 5 Å side-chain heavy-atom intermolecular contact detection.
2. Interface residue identification (Antibody vs Antigen).
3. Antibody CDR interface vs Framework interface filtering.
4. 5 Å side-chain heavy-atom intramolecular contact detection among interface residues.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
from scipy.spatial import cKDTree

from .constants import BACKBONE_ATOM_NAMES
from .models import ComplexData, ResidueData, AtomData


def get_sidechain_heavy_atoms(residue: ResidueData) -> List[AtomData]:
    """Return all side-chain heavy atoms of a residue.

    Strictly excludes:
    - Backbone atoms: N, CA, C, O, OXT
    - Hydrogen atoms: element 'H' or atom names starting with 'H'
    - Non-heavy or non-sidechain atoms
    """
    heavy_sidechain = []
    for a in residue.atoms:
        name_clean = a.name.strip().upper()
        elem_clean = (a.element or "").strip().upper()
        if name_clean in BACKBONE_ATOM_NAMES:
            continue
        if elem_clean == "H" or name_clean.startswith("H"):
            continue
        if a.is_sidechain and a.is_heavy:
            heavy_sidechain.append(a)
    return heavy_sidechain


@dataclass
class ContactRecord:
    """Detailed record of a qualifying atom-level contact."""
    antibody_chain: str
    antibody_res_id: str
    antibody_res_name: str
    antibody_res_num: int
    antibody_atom: str

    antigen_chain: str
    antigen_res_id: str
    antigen_res_name: str
    antigen_res_num: int
    antigen_atom: str

    distance: float

    def __str__(self) -> str:
        return (
            f"{self.antibody_res_id} {self.antibody_res_name}:{self.antibody_atom} ↔ "
            f"{self.antigen_res_id} {self.antigen_res_name}:{self.antigen_atom} = {self.distance:.2f} Å"
        )


@dataclass
class IntermolecularContactResult:
    """Result of intermolecular contact extraction."""
    contacts: List[ContactRecord]
    antibody_interface_residues: Dict[str, ResidueData]
    antigen_interface_residues: Dict[str, ResidueData]
    cdr_interface_residues: Dict[str, ResidueData]
    framework_interface_residues: Dict[str, ResidueData]

    @property
    def contact_count(self) -> int:
        return len(self.contacts)

    @property
    def antibody_interface_count(self) -> int:
        return len(self.antibody_interface_residues)

    @property
    def cdr_interface_count(self) -> int:
        return len(self.cdr_interface_residues)

    @property
    def framework_interface_count(self) -> int:
        return len(self.framework_interface_residues)

    @property
    def antigen_interface_count(self) -> int:
        return len(self.antigen_interface_residues)


def find_intermolecular_contacts(
    complex_data: ComplexData,
    cutoff: float = 5.0
) -> IntermolecularContactResult:
    """Extract intermolecular contacts between antibody and antigen side-chain heavy atoms.

    Distance rule: distance <= cutoff (5.0 Å).
    Excludes backbone atoms (N, CA, C, O, OXT) and hydrogen atoms.

    Args:
        complex_data: Fully parsed and entity-annotated complex.
        cutoff: Distance cutoff in Angstroms (default: 5.0).

    Returns:
        IntermolecularContactResult containing contact records and classified interface residues.
    """
    # 1. Collect antibody standard residues
    h_cid = complex_data.antibody.heavy_chain_id
    l_cid = complex_data.antibody.light_chain_id

    ab_residues: List[ResidueData] = []
    if h_cid and h_cid in complex_data.chains:
        ab_residues.extend(complex_data.chains[h_cid].standard_residues)
    if l_cid and l_cid in complex_data.chains:
        ab_residues.extend(complex_data.chains[l_cid].standard_residues)

    # 2. Collect antigen standard residues across all designated antigen chains
    ag_residues: List[ResidueData] = []
    for cid in complex_data.antigen_chain_ids:
        if cid in complex_data.chains:
            ag_residues.extend(complex_data.chains[cid].standard_residues)

    # 3. Extract side-chain heavy atoms using centralized filtering
    ab_atoms: List[Tuple[ResidueData, AtomData]] = [
        (r, a) for r in ab_residues for a in get_sidechain_heavy_atoms(r)
    ]
    ag_atoms: List[Tuple[ResidueData, AtomData]] = [
        (r, a) for r in ag_residues for a in get_sidechain_heavy_atoms(r)
    ]

    contacts: List[ContactRecord] = []
    ab_interface_map: Dict[str, ResidueData] = {}
    ag_interface_map: Dict[str, ResidueData] = {}

    if not ab_atoms or not ag_atoms:
        return IntermolecularContactResult(
            contacts=[],
            antibody_interface_residues={},
            antigen_interface_residues={},
            cdr_interface_residues={},
            framework_interface_residues={},
        )

    ab_coords = np.array([a.coord for _, a in ab_atoms], dtype=np.float32)
    ag_coords = np.array([a.coord for _, a in ag_atoms], dtype=np.float32)

    # Spatial query using cKDTree
    tree_ag = cKDTree(ag_coords)
    neighbor_indices = tree_ag.query_ball_point(ab_coords, r=cutoff)

    for ab_idx, hit_ag_indices in enumerate(neighbor_indices):
        r_ab, a_ab = ab_atoms[ab_idx]
        for ag_idx in hit_ag_indices:
            r_ag, a_ag = ag_atoms[ag_idx]
            dist = float(np.linalg.norm(a_ab.coord - a_ag.coord))
            if dist <= cutoff:
                contacts.append(ContactRecord(
                    antibody_chain=r_ab.chain_id,
                    antibody_res_id=r_ab.residue_id,
                    antibody_res_name=r_ab.residue_name,
                    antibody_res_num=r_ab.residue_number,
                    antibody_atom=a_ab.name,
                    antigen_chain=r_ag.chain_id,
                    antigen_res_id=r_ag.residue_id,
                    antigen_res_name=r_ag.residue_name,
                    antigen_res_num=r_ag.residue_number,
                    antigen_atom=a_ag.name,
                    distance=dist,
                ))
                ab_interface_map[r_ab.residue_id] = r_ab
                ag_interface_map[r_ag.residue_id] = r_ag

    # 4. Filter antibody CDR interface residues vs framework
    cdr_res_lookup: Dict[str, ResidueData] = {}
    for cdr_name, cdr in complex_data.antibody.cdrs.items():
        for r in cdr.residues:
            cdr_res_lookup[r.residue_id] = r

    cdr_interface_map: Dict[str, ResidueData] = {
        rid: res for rid, res in ab_interface_map.items() if rid in cdr_res_lookup
    }
    framework_interface_map: Dict[str, ResidueData] = {
        rid: res for rid, res in ab_interface_map.items() if rid not in cdr_res_lookup
    }

    return IntermolecularContactResult(
        contacts=contacts,
        antibody_interface_residues=ab_interface_map,
        antigen_interface_residues=ag_interface_map,
        cdr_interface_residues=cdr_interface_map,
        framework_interface_residues=framework_interface_map,
    )


@dataclass
class IntramolecularContactResult:
    """Result of intramolecular contact extraction among interface residues."""
    residue_pairs: List[Tuple[ResidueData, ResidueData]]
    pair_frequencies: Dict[Tuple[str, str], int]

    @property
    def total_contacts(self) -> int:
        return len(self.residue_pairs)

    @property
    def unique_pair_types(self) -> int:
        return len(self.pair_frequencies)


def find_intramolecular_contacts(
    residues: List[ResidueData],
    cutoff: float = 5.0
) -> IntramolecularContactResult:
    """Calculate intramolecular contacts among the given interface residues.

    Uses a 5.0 Å cutoff between side-chain heavy atoms.
    Each unique contacting residue pair (r1, r2) with r1 != r2 is counted ONCE,
    matching Biopython's NeighborSearch.search_all(level='R') in the author's code.

    Args:
        residues: List of interface ResidueData objects.
        cutoff: Distance cutoff in Angstroms (default: 5.0).

    Returns:
        IntramolecularContactResult with contacting residue pairs and 3-letter AA pair frequency counts.
    """
    if len(residues) < 2:
        return IntramolecularContactResult(residue_pairs=[], pair_frequencies={})

    atom_coords: List[np.ndarray] = []
    atom_to_res_idx: List[int] = []

    for r_idx, res in enumerate(residues):
        for atom in get_sidechain_heavy_atoms(res):
            atom_coords.append(atom.coord)
            atom_to_res_idx.append(r_idx)

    if not atom_coords:
        return IntramolecularContactResult(residue_pairs=[], pair_frequencies={})

    coords_arr = np.array(atom_coords, dtype=np.float32)
    tree = cKDTree(coords_arr)
    atom_pairs = tree.query_pairs(r=cutoff)

    # Filter to unique contacting residue pairs (r1 != r2)
    contacting_res_indices: Set[Tuple[int, int]] = set()
    for i, j in atom_pairs:
        r1_idx = atom_to_res_idx[i]
        r2_idx = atom_to_res_idx[j]
        if r1_idx != r2_idx:
            lo, hi = sorted((r1_idx, r2_idx))
            contacting_res_indices.add((lo, hi))

    ordered_pairs: List[Tuple[ResidueData, ResidueData]] = []
    pair_frequencies: Dict[Tuple[str, str], int] = {}

    for lo, hi in sorted(contacting_res_indices):
        r1 = residues[lo]
        r2 = residues[hi]
        ordered_pairs.append((r1, r2))

        # Standardized alphabetical amino acid combination key
        aa1 = r1.residue_name
        aa2 = r2.residue_name
        comb_key = tuple(sorted((aa1, aa2)))
        pair_frequencies[comb_key] = pair_frequencies.get(comb_key, 0) + 1

    return IntramolecularContactResult(
        residue_pairs=ordered_pairs,
        pair_frequencies=pair_frequencies,
    )
