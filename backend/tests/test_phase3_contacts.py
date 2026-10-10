"""Phase 3 Test Suite: 5 Å Contact Extraction & Interaction Representation.

Validates:
Test A — Side-chain heavy atoms: Backbone atoms and hydrogens are excluded.
Test B — 5 Å cutoff: Boundary at <= 5.0 Å is accepted; > 5.0 Å is rejected.
Test C — Intermolecular contacts: Only antibody <-> antigen residue contacts are intermolecular.
Test D — Interface residues: Every interface residue participates in >= 1 contact.
Test E — CDR filtering: CDR interface residues are a subset of antibody interface residues.
Test F — Antigen intramolecular contacts: Only antigen interface residues participate.
Test G — Antibody intramolecular contacts: Only CDR interface residues participate.
Test H — No atom-pair inflation: Multiple atom pairs between the same 2 residues count once.
Test I — Deterministic amino-acid mapping: All 20 amino acids map to fixed indices.
Test J — Matrix dimensions: Canonical (20, 20) and author CNN tensor (20, 21).
Test K — Normalization: Per-complex min-max normalization and zero-range handling.
Test L — Deterministic output: Identical inputs produce identical matrices and SHA256.
Test M — Four structures: Valid, non-identical representations for 1EJO, 1NBZ, 1KC5, 1DQJ.
"""

import numpy as np
import pytest
from pathlib import Path

from backend.app.structure import (
    AtomData,
    ResidueData,
    ChainData,
    ComplexData,
    AntibodyData,
    CDRData,
    BACKBONE_ATOM_NAMES,
    AMINO_3_CODES,
    AMINO_1_CODES,
    AA_INDEX_MAP,
    get_sidechain_heavy_atoms,
    find_intermolecular_contacts,
    find_intramolecular_contacts,
    build_upper_triangle,
    build_interaction_representation,
    analyze_biological_entities,
)


def _make_residue(
    chain_id: str,
    res_num: int,
    res_name: str,
    atoms: list,
) -> ResidueData:
    """Helper to construct synthetic residues for unit testing."""
    return ResidueData(
        chain_id=chain_id,
        residue_number=res_num,
        insertion_code=" ",
        residue_name=res_name,
        one_letter_code=res_name[0],
        is_standard=True,
        atoms=atoms,
    )


# -----------------------------------------------------------------------------
# Test A — Side-chain heavy atoms
# -----------------------------------------------------------------------------
def test_a_sidechain_heavy_atoms_exclusion():
    """Verify that backbone atoms (N, CA, C, O, OXT) and hydrogens are excluded."""
    atoms = [
        # Backbone atoms
        AtomData(name="N", element="N", coord=np.array([0, 0, 0]), is_sidechain=False, is_heavy=True),
        AtomData(name="CA", element="C", coord=np.array([1, 0, 0]), is_sidechain=False, is_heavy=True),
        AtomData(name="C", element="C", coord=np.array([2, 0, 0]), is_sidechain=False, is_heavy=True),
        AtomData(name="O", element="O", coord=np.array([3, 0, 0]), is_sidechain=False, is_heavy=True),
        AtomData(name="OXT", element="O", coord=np.array([4, 0, 0]), is_sidechain=False, is_heavy=True),
        # Hydrogen atoms
        AtomData(name="H", element="H", coord=np.array([0, 1, 0]), is_sidechain=False, is_heavy=False),
        AtomData(name="HA", element="H", coord=np.array([1, 1, 0]), is_sidechain=False, is_heavy=False),
        AtomData(name="HB2", element="H", coord=np.array([0, 2, 0]), is_sidechain=True, is_heavy=False),
        # Side-chain heavy atoms
        AtomData(name="CB", element="C", coord=np.array([1, 2, 0]), is_sidechain=True, is_heavy=True),
        AtomData(name="CG", element="C", coord=np.array([1, 3, 0]), is_sidechain=True, is_heavy=True),
        AtomData(name="CD1", element="C", coord=np.array([1, 4, 0]), is_sidechain=True, is_heavy=True),
        AtomData(name="NE2", element="N", coord=np.array([1, 5, 0]), is_sidechain=True, is_heavy=True),
    ]

    res = _make_residue("H", 50, "HIS", atoms)
    sidechain_heavies = get_sidechain_heavy_atoms(res)

    extracted_names = [a.name for a in sidechain_heavies]
    assert extracted_names == ["CB", "CG", "CD1", "NE2"]

    # Verify no backbone atoms leaked
    for bb in BACKBONE_ATOM_NAMES:
        assert bb not in extracted_names

    # Verify no hydrogens leaked
    for a in sidechain_heavies:
        assert a.element != "H"
        assert not a.name.startswith("H")


# -----------------------------------------------------------------------------
# Test B — 5 Å cutoff boundary
# -----------------------------------------------------------------------------
def test_b_cutoff_boundary_inclusion_and_exclusion():
    """Verify that a contact at exactly 5.0 Å is accepted and > 5.0 Å is rejected."""
    # Contact at exactly 5.0 Å
    ab_atom_50 = AtomData(name="CB", element="C", coord=np.array([0.0, 0.0, 0.0]), is_sidechain=True, is_heavy=True)
    ag_atom_50 = AtomData(name="CB", element="C", coord=np.array([5.0, 0.0, 0.0]), is_sidechain=True, is_heavy=True)

    res_ab = _make_residue("H", 31, "TYR", [ab_atom_50])
    res_ag = _make_residue("A", 10, "ALA", [ag_atom_50])

    complex_exact = ComplexData(
        pdb_id="TEST",
        source_path=Path("dummy.cif"),
        models_count=1,
        chains={
            "H": ChainData(chain_id="H", residues=[res_ab], chain_type="HEAVY"),
            "A": ChainData(chain_id="A", residues=[res_ag], chain_type="ANTIGEN"),
        },
        antibody=AntibodyData(heavy_chain_id="H", light_chain_id=None),
        antigen_chain_ids=["A"],
    )

    result_exact = find_intermolecular_contacts(complex_exact, cutoff=5.0)
    assert result_exact.contact_count == 1
    assert pytest.approx(result_exact.contacts[0].distance, 1e-4) == 5.0

    # Contact at 5.001 Å (just beyond cutoff)
    ag_atom_5001 = AtomData(name="CB", element="C", coord=np.array([5.001, 0.0, 0.0]), is_sidechain=True, is_heavy=True)
    res_ag_far = _make_residue("A", 10, "ALA", [ag_atom_5001])

    complex_far = ComplexData(
        pdb_id="TEST",
        source_path=Path("dummy.cif"),
        models_count=1,
        chains={
            "H": ChainData(chain_id="H", residues=[res_ab], chain_type="HEAVY"),
            "A": ChainData(chain_id="A", residues=[res_ag_far], chain_type="ANTIGEN"),
        },
        antibody=AntibodyData(heavy_chain_id="H", light_chain_id=None),
        antigen_chain_ids=["A"],
    )

    result_far = find_intermolecular_contacts(complex_far, cutoff=5.0)
    assert result_far.contact_count == 0


# -----------------------------------------------------------------------------
# Test C — Intermolecular contacts
# -----------------------------------------------------------------------------
def test_c_intermolecular_contact_classification():
    """Verify only antibody <-> antigen residue contacts are classified as intermolecular."""
    ab_atom1 = AtomData(name="CB", element="C", coord=np.array([0.0, 0.0, 0.0]), is_sidechain=True, is_heavy=True)
    ab_atom2 = AtomData(name="CB", element="C", coord=np.array([2.0, 0.0, 0.0]), is_sidechain=True, is_heavy=True)
    ag_atom1 = AtomData(name="CB", element="C", coord=np.array([4.0, 0.0, 0.0]), is_sidechain=True, is_heavy=True)

    res_ab1 = _make_residue("H", 31, "TYR", [ab_atom1])
    res_ab2 = _make_residue("H", 32, "SER", [ab_atom2])
    res_ag1 = _make_residue("P", 5, "ALA", [ag_atom1])

    complex_data = ComplexData(
        pdb_id="TEST",
        source_path=Path("dummy.cif"),
        models_count=1,
        chains={
            "H": ChainData(chain_id="H", residues=[res_ab1, res_ab2], chain_type="HEAVY"),
            "P": ChainData(chain_id="P", residues=[res_ag1], chain_type="ANTIGEN"),
        },
        antibody=AntibodyData(heavy_chain_id="H", light_chain_id=None),
        antigen_chain_ids=["P"],
    )

    result = find_intermolecular_contacts(complex_data, cutoff=5.0)

    # res_ab1 ↔ ag_atom1 is 4.0 Å (qualifying intermolecular)
    # res_ab2 ↔ ag_atom1 is 2.0 Å (qualifying intermolecular)
    # res_ab1 ↔ res_ab2 is 2.0 Å, but both are antibody -> must NOT be in intermolecular contacts
    for contact in result.contacts:
        assert contact.antibody_chain == "H"
        assert contact.antigen_chain == "P"
    assert result.contact_count == 2


# -----------------------------------------------------------------------------
# Test D — Interface residues
# -----------------------------------------------------------------------------
def test_d_interface_residue_participation():
    """Every reported interface residue must participate in at least one qualifying contact."""
    complex_data = analyze_biological_entities("1EJO")
    result = find_intermolecular_contacts(complex_data, cutoff=5.0)

    ab_res_in_contacts = {c.antibody_res_id for c in result.contacts}
    ag_res_in_contacts = {c.antigen_res_id for c in result.contacts}

    for res_id in result.antibody_interface_residues:
        assert res_id in ab_res_in_contacts

    for res_id in result.antigen_interface_residues:
        assert res_id in ag_res_in_contacts

    assert len(result.antibody_interface_residues) == len(ab_res_in_contacts)
    assert len(result.antigen_interface_residues) == len(ag_res_in_contacts)


# -----------------------------------------------------------------------------
# Test E — CDR filtering
# -----------------------------------------------------------------------------
def test_e_cdr_interface_filtering():
    """CDR interface residues must be a strict subset of antibody interface residues."""
    complex_data = analyze_biological_entities("1EJO")
    result = find_intermolecular_contacts(complex_data, cutoff=5.0)

    ab_iface = set(result.antibody_interface_residues.keys())
    cdr_iface = set(result.cdr_interface_residues.keys())
    fw_iface = set(result.framework_interface_residues.keys())

    # CDR interface is a subset of all antibody interface
    assert cdr_iface.issubset(ab_iface)
    # Framework interface is disjoint from CDR interface
    assert cdr_iface.isdisjoint(fw_iface)
    # Partition sum matches exactly
    assert len(cdr_iface) + len(fw_iface) == len(ab_iface)
    assert len(cdr_iface) > 0
    assert len(fw_iface) > 0


# -----------------------------------------------------------------------------
# Test F — Antigen intramolecular contacts
# -----------------------------------------------------------------------------
def test_f_antigen_intramolecular_contacts():
    """Only antigen interface residues participate in antigen intramolecular contacts."""
    complex_data = analyze_biological_entities("1EJO")
    rep = build_interaction_representation(complex_data)

    ag_iface_ids = set(rep.intermolecular.antigen_interface_residues.keys())

    for r1, r2 in rep.antigen_intramolecular.residue_pairs:
        assert r1.residue_id in ag_iface_ids
        assert r2.residue_id in ag_iface_ids
        assert r1.residue_id != r2.residue_id


# -----------------------------------------------------------------------------
# Test G — Antibody intramolecular contacts
# -----------------------------------------------------------------------------
def test_g_antibody_intramolecular_contacts():
    """Only antibody CDR interface residues participate in antibody intramolecular contacts."""
    complex_data = analyze_biological_entities("1EJO")
    rep = build_interaction_representation(complex_data)

    cdr_iface_ids = set(rep.intermolecular.cdr_interface_residues.keys())

    for r1, r2 in rep.antibody_intramolecular.residue_pairs:
        assert r1.residue_id in cdr_iface_ids
        assert r2.residue_id in cdr_iface_ids
        assert r1.residue_id != r2.residue_id


# -----------------------------------------------------------------------------
# Test H — No atom-pair inflation
# -----------------------------------------------------------------------------
def test_h_no_atom_pair_inflation():
    """Multiple atom pairs between the same 2 residues must increment pair frequency only once."""
    # TYR residue with 3 sidechain atoms
    tyr_atoms = [
        AtomData(name="CB", element="C", coord=np.array([0.0, 0.0, 0.0]), is_sidechain=True, is_heavy=True),
        AtomData(name="CG", element="C", coord=np.array([1.0, 0.0, 0.0]), is_sidechain=True, is_heavy=True),
        AtomData(name="CD1", element="C", coord=np.array([2.0, 0.0, 0.0]), is_sidechain=True, is_heavy=True),
    ]
    # LEU residue with 2 sidechain atoms located 3.0 Å away (all pairs <= 5.0 Å)
    leu_atoms = [
        AtomData(name="CB", element="C", coord=np.array([0.0, 3.0, 0.0]), is_sidechain=True, is_heavy=True),
        AtomData(name="CG", element="C", coord=np.array([1.0, 3.0, 0.0]), is_sidechain=True, is_heavy=True),
    ]

    r1 = _make_residue("H", 31, "TYR", tyr_atoms)
    r2 = _make_residue("H", 32, "LEU", leu_atoms)

    intra_result = find_intramolecular_contacts([r1, r2], cutoff=5.0)

    # 3 x 2 = 6 atom pairs satisfy the 5 Å cutoff, but there is only 1 residue pair
    assert intra_result.total_contacts == 1
    comb_key = tuple(sorted(("TYR", "LEU")))
    assert intra_result.pair_frequencies[comb_key] == 1


# -----------------------------------------------------------------------------
# Test I — Deterministic amino-acid mapping
# -----------------------------------------------------------------------------
def test_i_deterministic_amino_acid_mapping():
    """Verify deterministic 0..19 mapping matching the author's code and paper."""
    expected_order = [
        ("ALA", "A", 0),
        ("ARG", "R", 1),
        ("ASN", "N", 2),
        ("ASP", "D", 3),
        ("CYS", "C", 4),
        ("GLN", "Q", 5),
        ("GLU", "E", 6),
        ("GLY", "G", 7),
        ("HIS", "H", 8),
        ("ILE", "I", 9),
        ("LEU", "L", 10),
        ("LYS", "K", 11),
        ("MET", "M", 12),
        ("PHE", "F", 13),
        ("PRO", "P", 14),
        ("SER", "S", 15),
        ("THR", "T", 16),
        ("TRP", "W", 17),
        ("TYR", "Y", 18),
        ("VAL", "V", 19),
    ]

    assert len(AMINO_3_CODES) == 20
    assert len(AMINO_1_CODES) == 20

    for aa3, aa1, idx in expected_order:
        assert AMINO_3_CODES[idx] == aa3
        assert AMINO_1_CODES[idx] == aa1
        assert AA_INDEX_MAP[aa3] == idx


# -----------------------------------------------------------------------------
# Test J — Matrix dimensions
# -----------------------------------------------------------------------------
def test_j_matrix_dimensions():
    """Verify canonical (20, 20) matrix and author CNN tensor (20, 21) dimensions."""
    complex_data = analyze_biological_entities("1EJO")
    rep = build_interaction_representation(complex_data)

    # Canonical scientific representation
    assert rep.matrix_20x20.shape == (20, 20)
    assert rep.matrix_20x20.dtype == np.float32

    # Author's CNN input tensor
    assert rep.tensor_20x21.shape == (20, 21)
    assert rep.tensor_20x21.dtype == np.float32


# -----------------------------------------------------------------------------
# Test K — Per-complex normalization
# -----------------------------------------------------------------------------
def test_k_per_complex_normalization():
    """Verify min-max per-complex normalization and zero-range safe handling."""
    # Case 1: Standard frequencies
    counts = {
        ("ALA", "ALA"): 2,
        ("ALA", "TYR"): 8,
        ("LEU", "VAL"): 4,
    }
    raw, norm = build_upper_triangle(counts)
    assert np.nanmin(norm) == 0.0
    assert np.nanmax(norm) == 1.0
    assert norm[0, 0] == (2 - 0) / (8 - 0)
    assert norm[AA_INDEX_MAP["ALA"], AA_INDEX_MAP["TYR"]] == 1.0

    # Case 2: Populated frequencies with equal positive values
    counts_with_equal_positives = {
        ("ALA", "ALA"): 3,
        ("ALA", "TYR"): 3,
    }
    raw_id, norm_id = build_upper_triangle(counts_with_equal_positives)
    # The two present pairs scale to 1.0, while uncontacted pairs are 0.0
    assert norm_id[0, 0] == 1.0
    assert norm_id[AA_INDEX_MAP["ALA"], AA_INDEX_MAP["TYR"]] == 1.0
    assert norm_id[AA_INDEX_MAP["ARG"], AA_INDEX_MAP["ARG"]] == 0.0

    # Case 3: True zero-range edge case (all values 0, empty dictionary)
    raw_empty, norm_empty = build_upper_triangle({})
    for r in range(20):
        for c in range(r, 20):
            assert not np.isnan(norm_empty[r, c])
            assert norm_empty[r, c] == 0.0

    # Case 4: True zero-range edge case where every pair in the upper triangle has identical positive count
    all_five_counts = {}
    for r in range(20):
        for c in range(r, 20):
            all_five_counts[(AMINO_3_CODES[r], AMINO_3_CODES[c])] = 5
    raw_fives, norm_fives = build_upper_triangle(all_five_counts)
    # nan_max == nan_min == 5; must not divide by zero or yield NaNs
    for r in range(20):
        for c in range(r, 20):
            assert not np.isnan(norm_fives[r, c])
            assert norm_fives[r, c] == 0.0


# -----------------------------------------------------------------------------
# Test L — Deterministic output
# -----------------------------------------------------------------------------
def test_l_deterministic_output():
    """Running the pipeline twice on the same structure produces identical bytes and hashes."""
    comp1 = analyze_biological_entities("1EJO")
    rep1 = build_interaction_representation(comp1)

    comp2 = analyze_biological_entities("1EJO")
    rep2 = build_interaction_representation(comp2)

    assert np.array_equal(rep1.matrix_20x20, rep2.matrix_20x20)
    assert np.array_equal(rep1.tensor_20x21, rep2.tensor_20x21)
    assert rep1.stats_20x20.sha256 == rep2.stats_20x20.sha256
    assert rep1.stats_20x21.sha256 == rep2.stats_20x21.sha256


# -----------------------------------------------------------------------------
# Test M — Four benchmark structures
# -----------------------------------------------------------------------------
def test_m_four_benchmark_structures():
    """Validate representations for 1EJO, 1NBZ, 1KC5, 1DQJ and verify distinct hashes."""
    benchmarks = ["1EJO", "1NBZ", "1KC5", "1DQJ"]
    representations = {}
    hashes_20x20 = set()
    hashes_20x21 = set()

    for pdb_id in benchmarks:
        comp = analyze_biological_entities(pdb_id)
        rep = build_interaction_representation(comp)
        representations[pdb_id] = rep

        # Assert valid shapes
        assert rep.matrix_20x20.shape == (20, 20)
        assert rep.tensor_20x21.shape == (20, 21)

        # Assert biologically realistic contacts detected
        assert rep.intermolecular.contact_count > 50
        assert rep.intermolecular.antibody_interface_count > 5
        assert rep.intermolecular.cdr_interface_count > 5
        assert rep.intermolecular.antigen_interface_count > 5

        # Assert normalized bounds
        assert rep.stats_20x20.min_val >= 0.0
        assert rep.stats_20x20.max_val <= 1.0
        assert rep.stats_20x21.min_val >= 0.0
        assert rep.stats_20x21.max_val <= 1.0

        hashes_20x20.add(rep.stats_20x20.sha256)
        hashes_20x21.add(rep.stats_20x21.sha256)

    # All 4 benchmark structures MUST produce distinct, unique representations
    assert len(hashes_20x20) == 4, f"Found duplicate 20x20 hashes: {hashes_20x20}"
    assert len(hashes_20x21) == 4, f"Found duplicate 20x21 hashes: {hashes_20x21}"
