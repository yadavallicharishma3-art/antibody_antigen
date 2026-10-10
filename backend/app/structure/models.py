"""Internal data models for residues, chains, CDRs, antibodies, and complexes."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np


@dataclass
class AtomData:
    """Represents a parsed structural atom."""
    name: str
    element: str
    coord: np.ndarray
    is_sidechain: bool
    is_heavy: bool


@dataclass
class ResidueData:
    """Represents a parsed structural residue."""
    chain_id: str
    residue_number: int
    insertion_code: str = " "
    residue_name: str = ""
    one_letter_code: str = ""
    is_standard: bool = False
    atoms: List[AtomData] = field(default_factory=list)

    @property
    def residue_id(self) -> str:
        """Stable internal identifier, e.g. 'H:31' or 'H:52A'."""
        ins = self.insertion_code.strip()
        return f"{self.chain_id}:{self.residue_number}{ins}"

    @property
    def sidechain_heavy_atoms(self) -> List[AtomData]:
        """Returns side-chain heavy atoms (excluding backbone N, CA, C, O, OXT and hydrogens)."""
        return [a for a in self.atoms if a.is_sidechain and a.is_heavy]


@dataclass
class ChainData:
    """Represents a structural chain."""
    chain_id: str
    residues: List[ResidueData] = field(default_factory=list)
    chain_type: str = "UNKNOWN"  # "HEAVY", "LIGHT", "ANTIGEN", or "OTHER"
    evidence: str = ""

    @property
    def standard_residues(self) -> List[ResidueData]:
        return [r for r in self.residues if r.is_standard]

    @property
    def sequence(self) -> str:
        return "".join(r.one_letter_code for r in self.standard_residues)


@dataclass
class CDRData:
    """Represents a complementarity-determining region."""
    name: str  # e.g. "H1", "H2", "H3", "L1", "L2", "L3"
    chain_id: str
    sequence: str
    expected_sequence: Optional[str] = None
    residues: List[ResidueData] = field(default_factory=list)
    mapping_status: str = "PENDING"  # "MATCH", "PARTIAL", "FAILED"
    notes: str = ""

    @property
    def count(self) -> int:
        return len(self.residues)


@dataclass
class AntibodyData:
    """Represents the identified antibody components."""
    heavy_chain_id: Optional[str] = None
    light_chain_id: Optional[str] = None
    evidence: str = ""
    cdrs: Dict[str, CDRData] = field(default_factory=dict)


@dataclass
class ComplexData:
    """Represents the complete parsed and annotated antibody-antigen complex."""
    pdb_id: str
    source_path: str
    models_count: int
    chains: Dict[str, ChainData] = field(default_factory=dict)
    antibody: AntibodyData = field(default_factory=AntibodyData)
    antigen_chain_ids: List[str] = field(default_factory=list)
    antigen_evidence: str = ""
    warnings: List[str] = field(default_factory=list)
