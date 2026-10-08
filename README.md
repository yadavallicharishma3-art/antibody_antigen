# ANTIBODY–ANTIGEN INTERACTION ANALYZER V2
## Research Paper Reproduction + Full-Stack ML Application
**Reference Study**: *"Machine-learning-based Structural Analysis of Interactions between Antibodies and Antigens"* — Grace Zhang, Zhaoqian Su, Tom Zhang, Yinghao Wu (2024, *BioSystems* 243:105264; bioRxiv: 10.1101/2023.12.06.570397).

---

## 1. Project Overview & Scientific Source of Truth

This project reproduces the core methodology from Zhang et al. (2024) to computationally analyze structural interaction patterns at antibody-antigen interfaces and evaluate them against general protein-protein interactions (PPI) using a convolutional neural network (CNN).

### Key Scientific Principles
1. **Intermolecular Contacts (5 Å side-chain heavy atoms)**:
   - Identify which residues are on the interaction interface (antibody CDR residues ↔ antigen residues).
2. **Intramolecular Contacts (5 Å side-chain heavy atoms)**:
   - Calculate internal contacts **between** interface residues within the antigen (upper diagonal).
   - Calculate internal contacts **between** interface residues within the antibody CDRs (lower diagonal).
   - Intermolecular contacts are **never** placed directly into the 20×20 matrix.
3. **Per-Complex Normalization**:
   - Each complex is individually min-max normalized between 0 and 1.
4. **CNN Architecture & 288-Dimensional Tensor**:
   - 3× Conv2D(32 filters, 3×3 kernel, ReLU, padding='same') followed by MaxPooling2D((2,2), padding='same').
   - Produces a feature map of shape `(3, 3, 32)` which flattens to **exactly 288 dimensions**.
   - Dense(512) → Dense(64) → Dense(1, sigmoid).

---

## 2. Phase 1 Audit & Scientific Findings

### Dataset Discovery & Schema
| Dataset | Source | Total Items | Format & Schema |
| :--- | :--- | :--- | :--- |
| **AbDb / SAbDab** | AbDb / SAbDab export | 1,215 complexes (`AbDb_list.dat`) | PDB/CIF structures; `abag_split.csv` containing IMGT numbering, CDR H1-H3 & L1-L3 ranges, heavy/light chain IDs, and antigen chain IDs. |
| **3did General PPI** | 3did Database | 4,144 non-redundant pairs (`pdb_4960list.txt`) | Tab-delimited: `Index PDB Chain1 Start1 End1 Chain2 Start2 End2` (3,218 inter-chain pairs used in Test 1). |

### 288-Dimensional Bottleneck Resolution
The paper specifies a 288-dimensional flattened representation. Our mathematical and code audit of the author's reference notebook revealed:
- When using `MaxPooling2D((2, 2))` with default `padding='valid'`, the output dimensions collapse: `(20, 20) -> (10, 10) -> (5, 5) -> (2, 2)`, yielding `2 * 2 * 32 = 128` dimensions (the flaw in the previous reference implementation).
- When using `MaxPooling2D((2, 2), padding='same')`, the odd spatial dimension is preserved: `ceil(5/2) = 3`, yielding `3 * 3 * 32 = 288` dimensions.
- Test `backend/tests/test_phase1_environment.py::test_cnn_architecture_produces_exact_288_dimensions` verified this behavior.

---

## 3. Directory Layout

```
ANTIBODY-ANTIGEN-ANALYZER-V2/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── datasets/
│   │   ├── ml/
│   │   ├── schemas/
│   │   │   └── analysis.py
│   │   └── structure/
│   ├── tests/
│   │   └── test_phase1_environment.py
│   └── requirements.txt
├── data/
│   ├── abdb/
│   │   ├── AbDb_list.dat
│   │   └── abag_split.csv
│   ├── 3did/
│   │   └── pdb_4960list.txt
│   ├── structures/
│   │   ├── 1EJO.cif
│   │   ├── 1NBZ.cif
│   │   ├── 1KC5.cif
│   │   └── 1DQJ.cif (+ 58 cached CIF files)
│   └── processed/
├── model/
├── scripts/
├── experiments/
│   ├── test1/
│   ├── test2/
│   └── test3/
└── README.md
```

---

## 4. Phase 1 & 2 Verification Results

### Phase 1 Tests (6/6 Passed)
- `test_core_dependencies_importable` (PASSED)
- `test_abdb_dataset_list` (PASSED)
- `test_3did_control_dataset_list` (PASSED)
- `test_sabdab_metadata_csv` (PASSED)
- `test_required_structures_present` (PASSED)
- `test_cnn_architecture_produces_exact_288_dimensions` (PASSED)

### Phase 2 Tests (18/18 Passed)
- `test_a_load_known_structures[1EJO, 1NBZ, 1KC5, 1DQJ]` (4 PASSED)
- `test_b_chain_extraction[1EJO, 1NBZ, 1KC5, 1DQJ]` (4 PASSED)
- `test_c_standard_residues_and_mapping` (PASSED)
- `test_d_residue_identity_and_insertion_codes` (PASSED)
- `test_e_antibody_chain_identification` (PASSED - verified Heavy=B, Light=A for 1NBZ/1DQJ, no hardcoding)
- `test_f_antigen_chain_identification` (PASSED)
- `test_g_six_cdr_regions_present[1EJO, 1NBZ, 1KC5, 1DQJ]` (4 PASSED - all 6 CDRs match expected sequences)
- `test_h_cdr_to_structure_mapping` (PASSED - mapped residues have valid atomic coordinates)
- `test_i_ambiguity_handling` (PASSED - unannotated structures do not silently assign chains)

---

## 5. Phase 2 Structure Parsing & Entity Identification Architecture

### Supported Formats & Sources
- Formats: `.cif`, `.mmcif`, `.pdb`
- Retrieval: Local cache in `data/structures/`, with automated fallback retrieval from RCSB PDB (`https://files.rcsb.org/download/{pdb_id}.cif`).

### Chain & Entity Identification Method
- **Antibody Heavy and Light Chains**: Identified from `abag_split.csv` (`Hchain`, `Lchain`) with recorded biological evidence. Avoids hardcoded assumptions (e.g. In `1NBZ` and `1DQJ`, Heavy chain is `B` and Light chain is `A`).
- **Antigen Chains**: Sourced from `agchains` metadata, handling multi-chain antigens (e.g. `A/B`).
- **Residue Numbering & Identity**: Preserves original crystallographic residue numbering (e.g. `1EJO` Chain H numbered in the 2500s) and insertion codes without arbitrary renumbering.
- **CDR Identification**: Aligns structural standard residues against SAbDab IMGT numbering sequences using global pairwise alignment. Mapped residues are extracted within IMGT standard intervals:
  - CDR1: 27–38
  - CDR2: 56–65
  - CDR3: 105–117
  Mapped CDR sequences are cross-validated against expected sequences (`CDRH1..3`, `CDRL1..3`).

### Diagnostic CLI
Run the diagnostic tool for any structure:
```bash
# Structure and entity diagnostics (Phase 2)
python scripts/analyze_structure.py 1EJO --phase structure

# Matrix and contact diagnostics (Phase 3)
python scripts/analyze_structure.py 1EJO --phase matrix -v

# Run full diagnostic and export matrix files (.npy and .csv)
python scripts/analyze_structure.py 1EJO --export
```

---

## 6. Phase 3 5 Å Contact Extraction & Representation Architecture

### Scientific Methodology
1. **Side-Chain Heavy Atom Filtering**:
   - Strictly excludes backbone atoms: `N`, `CA`, `C`, `O`, `OXT`.
   - Strictly excludes hydrogen atoms: element `H` or names starting with `H`.
   - Residues without side-chain heavy atoms (e.g. standard Glycine) produce no side-chain contacts.
2. **Intermolecular Contacts (<= 5.0 Å Cutoff)**:
   - Uses `scipy.spatial.cKDTree` for spatial neighbor search between antibody side-chain heavy atoms and antigen side-chain heavy atoms.
   - Identifies interface residues:
     - All antibody interface residues.
     - CDR interface residues (subset belonging to IMGT CDR H1–H3 or L1–L3).
     - Framework interface residues.
     - Antigen interface residues (supporting multi-chain antigens).
3. **Intramolecular Interface Contacts**:
   - **Antigen**: Contacts calculated strictly among antigen interface residues.
   - **Antibody**: Contacts calculated strictly among antibody CDR interface residues.
   - **No Atom-Pair Multiplicity Inflation**: Each contacting residue pair $(r_1, r_2)$ with $r_1 \neq r_2$ is counted exactly once, matching Biopython's `NeighborSearch.search_all(5, level='R')` in author's reference notebook `part1.ipynb`.
4. **Deterministic Amino Acid Mapping**:
   - 20 standard amino acids alphabetized by 3-letter code:
     `ALA (0), ARG (1), ASN (2), ASP (3), CYS (4), GLN (5), GLU (6), GLY (7), HIS (8), ILE (9), LEU (10), LYS (11), MET (12), PHE (13), PRO (14), SER (15), THR (16), TRP (17), TYR (18), VAL (19)`.
   - Corresponding 1-letter codes: `A, R, N, D, C, Q, E, G, H, I, L, K, M, F, P, S, T, W, Y, V`.
5. **Upper / Lower Triangle Construction & Orientation**:
   - Antigen intramolecular pair frequencies populate the upper triangle (`r <= c`), with below-diagonal as `NaN`.
   - Antibody CDR intramolecular pair frequencies populate the upper triangle of the antibody chart, with below-diagonal as `NaN`.
   - Diagonal entries represent contacts between different residues of the same amino-acid type (e.g. `TYR` ↔ `TYR`).
6. **Per-Complex Min-Max Normalization**:
   - Formula: $x' = \frac{x - \text{nanmin}(x)}{\text{nanmax}(x) - \text{nanmin}(x)}$
   - Evaluated independently per complex (never global across the dataset).
   - Safe zero-range handling: If $\text{nanmax} = \text{nanmin}$ (e.g. empty or constant), populates with $0.0$ to prevent division by zero or NaN generation.
7. **Matrix Dimensions (20×20 vs 20×21)**:
   - **Canonical Scientific Representation `(20, 20)`**: Upper triangle holds antigen normalized contacts; lower triangle holds antibody CDR normalized contacts.
   - **Author's CNN Input Tensor `(20, 21)`**: Flipping the antibody upper triangle (`np.flip(Ab)`) yields $r + 1$ non-NaN values in row $r$; concatenating with antigen upper triangle row $r$ ($20 - r$ non-NaN values) and filtering out NaNs yields exactly $21$ values per row for all $20$ rows. Input shape to CNN is `(20, 21, 1)`.
   - Both representations are computed, tested, and exported.

### Phase 3 Benchmark Diagnostics
| Benchmark PDB | Intermolecular Contacts | Ab Interface (CDR / FW) | Ag Interface | Ab Intra (CDR) | Ag Intra | 20×20 SHA256 | 20×21 SHA256 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1EJO** | 233 | 22 (17 / 5) | 11 | 24 | 10 | `0fb0762e2c28...` | `8d505d81e339...` |
| **1NBZ** | 278 | 21 (17 / 4) | 16 | 22 | 12 | `658edd5d6a38...` | `eabd0045cffb...` |
| **1KC5** | 120 | 12 (10 / 2) | 6 | 6 | 5 | `03bcd08cc699...` | `f1f3d2218b68...` |
| **1DQJ** | 315 | 20 (16 / 4) | 16 | 21 | 14 | `e9ab5a4cdc45...` | `43a1a031a9ca...` |

### Phase 3 Test Suite (13/13 Passed; Total 37/37 across Phases 1–3)
- `test_a_sidechain_heavy_atoms_exclusion` (PASSED)
- `test_b_cutoff_boundary_inclusion_and_exclusion` (PASSED)
- `test_c_intermolecular_contact_classification` (PASSED)
- `test_d_interface_residue_participation` (PASSED)
- `test_e_cdr_interface_filtering` (PASSED)
- `test_f_antigen_intramolecular_contacts` (PASSED)
- `test_g_antibody_intramolecular_contacts` (PASSED)
- `test_h_no_atom_pair_inflation` (PASSED)
- `test_i_deterministic_amino_acid_mapping` (PASSED)
- `test_j_matrix_dimensions` (PASSED)
- `test_k_per_complex_normalization` (PASSED)
- `test_l_deterministic_output` (PASSED)
- `test_m_four_benchmark_structures` (PASSED)

---

## 7. Phase 4 Dataset Construction & Leakage-Safe Validation

### Dataset Architecture & Filtering Pipeline
1. **Metadata Source**:
   - `data/abdb/abag_split.csv`: 15,641 instances representing 8,641 unique PDB complexes with SAbDab IMGT numbering, sequence clusters (`ab_cluster`, `agclusters`, `ab_ag_cluster`), and official benchmark split (`ab_ag_split`).
2. **Deterministic Rejection Categorization**:
   - Samples undergo strict biological and structural validation. No zero-matrix fallbacks or synthetic perturbations are allowed.
   - Rejection categories:
     - `missing_pdb_id`
     - `missing_structure`
     - `structure_parse_error`
     - `missing_antibody_metadata`
     - `missing_antigen_metadata` (e.g. unbound structures or hapten complexes)
     - `cdr_mapping_failure`
     - `no_interface_contacts`
     - `no_valid_representation`
     - `unsupported_structure`
3. **Leakage-Safe Cluster Partitioning**:
   - Preserves SAbDab's official test partition isolated strictly by `ab_ag_cluster`.
   - Sub-partitions training candidates into `train` (~80%) and `validation` (~20%) using group-splitting on `ab_ag_cluster` with fixed seed 42.
   - Zero Ab-Ag cluster overlap and zero PDB overlap between test and train/val sets.
4. **Reproducible Storage**:
   - Manifest: `data/processed/dataset_manifest.csv`
   - Rejections: `data/processed/rejected_samples.csv`
   - Split manifests: `data/processed/train.csv`, `data/processed/validation.csv`, `data/processed/test.csv`
   - Author-compatible tensors: `data/processed/representations/<sample_id>.npy` (shape `(20, 21)`).
5. **Unseen Structure Validation**:
   - Structure `1A3R` (not part of the Phase 3 benchmarks) was executed through the identical production pipeline:
     - Heavy: `H`, Light: `L`, Antigen: `['P']`
     - All 6 CDRs matched to coordinates
     - 373 intermolecular contacts, 23 CDR interface residues, 14 Ag interface residues
     - 28 Ab CDR intra contacts, 15 Ag intra contacts
     - Valid (20, 21) tensor, 30 nonzero cells, SHA256: `c422403f6c9c7978...`

### Build Script CLI
```bash
# Build dataset on all cached structures with deterministic splitting
python scripts/build_dataset.py

# Re-run with custom validation fraction or seed
python scripts/build_dataset.py --val-fraction 0.20 --seed 42 --unseen-pdb 1A3R
```

### Phase 4 Test Suite (12/12 Passed; Combined 49/49 across Phases 1–4)
- `test_a_manifest_schema_is_valid` (PASSED)
- `test_b_every_accepted_sample_has_representation` (PASSED)
- `test_c_representation_shapes` (PASSED)
- `test_d_no_nans_or_infs` (PASSED)
- `test_e_representation_hashes_are_deterministic` (PASSED)
- `test_f_rejected_samples_have_explicit_reasons` (PASSED)
- `test_g_pdb_ids_do_not_overlap_unexpectedly` (PASSED)
- `test_h_cluster_leakage_checks` (PASSED)
- `test_i_split_generation_is_deterministic` (PASSED)
- `test_j_unseen_structure_pipeline` (PASSED)
- `test_k_rebuilding_dataset_produces_same_manifests` (PASSED)
- `test_l_no_benchmark_specific_hardcoded_processing` (PASSED)



