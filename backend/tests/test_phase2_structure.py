"""Phase 2 Structure Parsing and Biological Entity Identification Test Suite.

Covers Tests A through I as specified in the Phase 2 requirements:
- Test A: Load structures
- Test B: Chain extraction
- Test C: Standard residue extraction & 20 amino acid mapping
- Test D: Residue identity & insertion code preservation
- Test E: Antibody chain identification (Heavy/Light with evidence, no hardcoding)
- Test F: Antigen chain identification (explicit identification with evidence)
- Test G: Six CDR regions presence (H1, H2, H3, L1, L2, L3)
- Test H: CDR-to-structure mapping (mapped residues exist in structure with atoms)
- Test I: Ambiguity handling (missing/unannotated metadata does not silently guess)
"""

from pathlib import Path
import pytest

from backend.app.structure import (
    resolve_structure_path,
    load_and_parse_complex,
    analyze_biological_entities,
    parse_structure_file,
    identify_complex_entities,
    identify_antibody_cdrs,
    generate_chain_inventory,
    STANDARD_AA_3TO1,
    STANDARD_AA_LETTERS,
    StructureLoadError,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
STRUCTURES_DIR = BASE_DIR / "data" / "structures"
TEST_PDBS = ["1EJO", "1NBZ", "1KC5", "1DQJ"]


# --- Test A: Load structures ---
@pytest.mark.parametrize("pdb_id", TEST_PDBS)
def test_a_load_known_structures(pdb_id):
    """Verify that all four benchmark structures can be located and loaded."""
    path = resolve_structure_path(pdb_id, allow_download=False)
    assert path.exists(), f"Structure file for {pdb_id} must exist locally."
    assert path.stat().st_size > 1000, f"Structure file for {pdb_id} is suspiciously small."


# --- Test B: Chain extraction ---
@pytest.mark.parametrize("pdb_id", TEST_PDBS)
def test_b_chain_extraction(pdb_id):
    """Verify that chains are correctly extracted into models."""
    complex_data = load_and_parse_complex(pdb_id)
    assert complex_data.chains, f"No chains extracted for {pdb_id}"
    assert len(complex_data.chains) >= 2, f"Expected at least 2 chains for complex {pdb_id}"
    for cid, chain in complex_data.chains.items():
        assert len(chain.residues) > 0, f"Chain {cid} in {pdb_id} has zero residues"


# --- Test C: Standard residue extraction & 20 amino acids ---
def test_c_standard_residues_and_mapping():
    """Verify standard amino acid dictionary and standard filtering."""
    assert len(STANDARD_AA_LETTERS) == 20
    assert len(STANDARD_AA_3TO1) == 20

    complex_data = load_and_parse_complex("1EJO")
    h_chain = complex_data.chains["H"]
    for res in h_chain.standard_residues:
        assert res.is_standard is True
        assert res.one_letter_code in STANDARD_AA_LETTERS
        assert len(res.one_letter_code) == 1


# --- Test D: Residue identity & insertion code preservation ---
def test_d_residue_identity_and_insertion_codes():
    """Verify residue numbers and insertion codes are preserved without renumbering."""
    complex_data = load_and_parse_complex("1EJO")
    # In 1EJO crystallographic CIF, Chain H residues are numbered in the 2500s
    h_chain = complex_data.chains["H"]
    res_numbers = [r.residue_number for r in h_chain.residues]
    assert 2501 in res_numbers, "Expected original crystallographic residue number 2501 in 1EJO Chain H"
    assert res_numbers[0] == 2501, "Should preserve original residue numbering, not 1-indexed renumbering"

    for r in h_chain.residues:
        assert isinstance(r.insertion_code, str)
        assert r.residue_id.startswith("H:")


# --- Test E: Antibody chain identification (evidence-based, no hardcoding) ---
def test_e_antibody_chain_identification():
    """Verify antibody heavy and light chains are identified from metadata.

    Crucially tests that 1NBZ and 1DQJ identify B as Heavy and A as Light,
    proving no hardcoding of H=Heavy and L=Light.
    """
    c_1ejo = analyze_biological_entities("1EJO")
    assert c_1ejo.antibody.heavy_chain_id == "H"
    assert c_1ejo.antibody.light_chain_id == "L"
    assert "SAbDab" in c_1ejo.antibody.evidence

    c_1nbz = analyze_biological_entities("1NBZ")
    assert c_1nbz.antibody.heavy_chain_id == "B", "1NBZ Heavy chain MUST be B (from metadata)"
    assert c_1nbz.antibody.light_chain_id == "A", "1NBZ Light chain MUST be A (from metadata)"
    assert "SAbDab" in c_1nbz.antibody.evidence

    c_1dqj = analyze_biological_entities("1DQJ")
    assert c_1dqj.antibody.heavy_chain_id == "B", "1DQJ Heavy chain MUST be B (from metadata)"
    assert c_1dqj.antibody.light_chain_id == "A", "1DQJ Light chain MUST be A (from metadata)"


# --- Test F: Antigen chain identification ---
def test_f_antigen_chain_identification():
    """Verify antigen chains are explicitly identified and preserved."""
    c_1ejo = analyze_biological_entities("1EJO")
    assert c_1ejo.antigen_chain_ids == ["P"]
    assert "agchains=P" in c_1ejo.antigen_evidence

    c_1nbz = analyze_biological_entities("1NBZ")
    assert c_1nbz.antigen_chain_ids == ["C"]
    assert "agchains=C" in c_1nbz.antigen_evidence

    c_1kc5 = analyze_biological_entities("1KC5")
    assert c_1kc5.antigen_chain_ids == ["P"]


# --- Test G: Six CDR regions presence ---
@pytest.mark.parametrize("pdb_id", TEST_PDBS)
def test_g_six_cdr_regions_present(pdb_id):
    """Verify that all six CDRs (H1, H2, H3, L1, L2, L3) are mapped for each complex."""
    complex_data = analyze_biological_entities(pdb_id)
    cdrs = complex_data.antibody.cdrs
    for cdr_name in ["H1", "H2", "H3", "L1", "L2", "L3"]:
        assert cdr_name in cdrs, f"CDR {cdr_name} missing from {pdb_id}"
        assert cdrs[cdr_name].count > 0, f"CDR {cdr_name} in {pdb_id} has 0 residues"
        assert cdrs[cdr_name].mapping_status == "MATCH", (
            f"CDR {cdr_name} in {pdb_id} did not match expected sequence: {cdrs[cdr_name].sequence} != {cdrs[cdr_name].expected_sequence}"
        )


# --- Test H: CDR-to-structure mapping ---
def test_h_cdr_to_structure_mapping():
    """Verify mapped CDR residues are real parsed structure residues with valid atoms."""
    complex_data = analyze_biological_entities("1EJO")
    h3_cdr = complex_data.antibody.cdrs["H3"]
    assert h3_cdr.sequence == "VRRAFDSDVGFAS"
    assert len(h3_cdr.residues) == 13

    for res in h3_cdr.residues:
        assert res.chain_id == "H"
        assert res.is_standard is True
        assert len(res.atoms) > 0, f"Residue {res.residue_id} must contain parsed atoms"
        sidechain_heavy = res.sidechain_heavy_atoms
        # Non-glycine residues in H3 must have sidechain heavy atoms
        if res.one_letter_code != "G":
            assert len(sidechain_heavy) > 0, f"Residue {res.residue_id} ({res.one_letter_code}) should have sidechain heavy atoms"


# --- Test I: Ambiguity handling ---
def test_i_ambiguity_handling():
    """Verify that when metadata is absent, the system does not make silent arbitrary guesses."""
    # Test non-existent PDB ID in loader
    with pytest.raises(StructureLoadError):
        resolve_structure_path("ZZZZ", allow_download=False)

    # Test parser on a structure without metadata
    c_1ejo = load_and_parse_complex("1EJO")
    c_unannotated = identify_complex_entities(c_1ejo, allow_metadata_lookup=False)
    # Without metadata, heavy and light chains must NOT be arbitrarily set
    assert c_unannotated.antibody.heavy_chain_id is None
    assert c_unannotated.antibody.light_chain_id is None
    assert any("No SAbDab metadata found" in w for w in c_unannotated.warnings)
