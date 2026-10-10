"""Dataset construction and structural filtering pipeline."""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd

from backend.app.structure import (
    load_and_parse_complex,
    identify_complex_entities,
    identify_antibody_cdrs,
    find_intermolecular_contacts,
    build_interaction_representation,
    resolve_structure_path,
    normalize_pdb_id,
    StructureLoadError,
    StructureParseError,
    ComplexData,
)
from .models import DatasetSample, RejectedSample

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_METADATA_PATH = BASE_DIR / "data" / "abdb" / "abag_split.csv"
DEFAULT_STRUCTURES_DIR = BASE_DIR / "data" / "structures"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "processed"
DEFAULT_REPRESENTATIONS_DIR = DEFAULT_OUTPUT_DIR / "representations"


class DatasetBuilder:
    """Orchestrates deterministic dataset construction, structural filtering, and tensor generation."""

    def __init__(
        self,
        metadata_path: Optional[Path] = None,
        structures_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        allow_download: bool = False,
    ):
        self.metadata_path = metadata_path or DEFAULT_METADATA_PATH
        self.structures_dir = structures_dir or DEFAULT_STRUCTURES_DIR
        self.output_dir = output_dir or DEFAULT_OUTPUT_DIR
        self.representations_dir = self.output_dir / "representations"
        self.allow_download = allow_download

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.representations_dir.mkdir(parents=True, exist_ok=True)

    def process_record(
        self,
        record: pd.Series,
    ) -> Tuple[Optional[DatasetSample], Optional[RejectedSample]]:
        """Process a single SAbDab metadata record through the complete structural pipeline.

        Returns:
            Tuple of (DatasetSample, None) if accepted, or (None, RejectedSample) if rejected.
        """
        instance_id = str(record.get("INSTANCE", "")).strip()
        raw_pdb = str(record.get("PDB_ID", "")).strip()
        clean_pdb = normalize_pdb_id(raw_pdb)

        if not clean_pdb or len(clean_pdb) != 4:
            return None, RejectedSample(
                sample_id=instance_id or "UNKNOWN",
                pdb_id=clean_pdb,
                reason="missing_pdb_id",
                detailed_error=f"Invalid PDB ID '{raw_pdb}'",
                stage_failed="metadata_validation",
            )

        sample_id = instance_id if instance_id else f"{clean_pdb}_default"

        # 1. Validate basic antibody and antigen metadata
        h_chain = str(record.get("Hchain", "")).strip() if not pd.isna(record.get("Hchain")) else ""
        l_chain = str(record.get("Lchain", "")).strip() if not pd.isna(record.get("Lchain")) else ""
        ag_chains_raw = str(record.get("agchains", "")).strip() if not pd.isna(record.get("agchains")) else ""

        if not h_chain and not l_chain:
            return None, RejectedSample(
                sample_id=sample_id,
                pdb_id=clean_pdb,
                reason="missing_antibody_metadata",
                detailed_error="Both Hchain and Lchain are empty in SAbDab metadata",
                stage_failed="metadata_validation",
            )

        if not ag_chains_raw:
            return None, RejectedSample(
                sample_id=sample_id,
                pdb_id=clean_pdb,
                reason="missing_antigen_metadata",
                detailed_error="No antigen chains specified in SAbDab metadata (agchains is empty)",
                stage_failed="metadata_validation",
            )

        # 2. Acquire and verify structure file
        try:
            struct_path = resolve_structure_path(clean_pdb, allow_download=self.allow_download)
        except StructureLoadError as err:
            return None, RejectedSample(
                sample_id=sample_id,
                pdb_id=clean_pdb,
                reason="missing_structure",
                detailed_error=str(err),
                stage_failed="structure_loading",
            )

        # 3. Parse structure
        try:
            complex_data = load_and_parse_complex(struct_path)
        except StructureParseError as err:
            return None, RejectedSample(
                sample_id=sample_id,
                pdb_id=clean_pdb,
                reason="structure_parse_error",
                detailed_error=str(err),
                stage_failed="structure_parsing",
            )
        except Exception as exc:
            return None, RejectedSample(
                sample_id=sample_id,
                pdb_id=clean_pdb,
                reason="structure_parse_error",
                detailed_error=f"Unexpected parsing error: {exc}",
                stage_failed="structure_parsing",
            )

        # 4. Identify complex entities using this specific record
        complex_data = identify_complex_entities(complex_data, sabdab_record=record)

        # Check if antibody chains are physically present
        has_h = bool(complex_data.antibody.heavy_chain_id and complex_data.antibody.heavy_chain_id in complex_data.chains)
        has_l = bool(complex_data.antibody.light_chain_id and complex_data.antibody.light_chain_id in complex_data.chains)
        if not has_h and not has_l:
            return None, RejectedSample(
                sample_id=sample_id,
                pdb_id=clean_pdb,
                reason="missing_antibody_metadata",
                detailed_error=f"Designated antibody chains (H={h_chain}, L={l_chain}) not found in parsed structure.",
                stage_failed="entity_identification",
            )

        # Check if designated antigen chains are physically present
        if not complex_data.antigen_chain_ids:
            return None, RejectedSample(
                sample_id=sample_id,
                pdb_id=clean_pdb,
                reason="missing_antigen_metadata",
                detailed_error=f"Designated antigen chains '{ag_chains_raw}' not found in parsed structure.",
                stage_failed="entity_identification",
            )

        # 5. Identify CDRs
        complex_data = identify_antibody_cdrs(complex_data, sabdab_record=record)
        valid_cdrs = [cdr for cdr in complex_data.antibody.cdrs.values() if cdr.mapping_status in ["MATCH", "PARTIAL"] and cdr.residues]
        if not valid_cdrs:
            return None, RejectedSample(
                sample_id=sample_id,
                pdb_id=clean_pdb,
                reason="cdr_mapping_failure",
                detailed_error="Could not map any CDR residues to atomic coordinates in structure.",
                stage_failed="cdr_mapping",
            )

        # 6. Intermolecular contact extraction
        inter_result = find_intermolecular_contacts(complex_data, cutoff=5.0)
        if inter_result.contact_count == 0:
            return None, RejectedSample(
                sample_id=sample_id,
                pdb_id=clean_pdb,
                reason="no_interface_contacts",
                detailed_error="Zero 5 Å side-chain heavy-atom intermolecular contacts detected between antibody and antigen.",
                stage_failed="contact_extraction",
            )

        if inter_result.cdr_interface_count == 0:
            return None, RejectedSample(
                sample_id=sample_id,
                pdb_id=clean_pdb,
                reason="no_interface_contacts",
                detailed_error="No intermolecular contacts involve antibody CDR residues (only framework contacts).",
                stage_failed="contact_extraction",
            )

        if inter_result.antigen_interface_count == 0:
            return None, RejectedSample(
                sample_id=sample_id,
                pdb_id=clean_pdb,
                reason="no_interface_contacts",
                detailed_error="No antigen residues participate in intermolecular contacts.",
                stage_failed="contact_extraction",
            )

        # 7. Generate structural representation
        try:
            rep = build_interaction_representation(complex_data, cutoff=5.0)
        except Exception as exc:
            return None, RejectedSample(
                sample_id=sample_id,
                pdb_id=clean_pdb,
                reason="no_valid_representation",
                detailed_error=f"Representation generation failed: {exc}",
                stage_failed="representation_generation",
            )

        # Verify representation validity
        if rep.tensor_20x21.shape != (20, 21):
            return None, RejectedSample(
                sample_id=sample_id,
                pdb_id=clean_pdb,
                reason="no_valid_representation",
                detailed_error=f"Unexpected tensor shape {rep.tensor_20x21.shape}; expected (20, 21).",
                stage_failed="representation_generation",
            )

        if np.isnan(rep.tensor_20x21).any() or np.isinf(rep.tensor_20x21).any():
            return None, RejectedSample(
                sample_id=sample_id,
                pdb_id=clean_pdb,
                reason="no_valid_representation",
                detailed_error="Tensor contains NaN or Inf values after normalization.",
                stage_failed="representation_generation",
            )

        # 8. Save author-compatible tensor to disk
        tensor_dest = self.representations_dir / f"{sample_id}.npy"
        np.save(tensor_dest, rep.tensor_20x21)

        sample = DatasetSample(
            sample_id=sample_id,
            pdb_id=clean_pdb,
            structure_path=str(struct_path.relative_to(BASE_DIR) if struct_path.is_relative_to(BASE_DIR) else struct_path),
            antibody_heavy_chain=complex_data.antibody.heavy_chain_id or "",
            antibody_light_chain=complex_data.antibody.light_chain_id or "",
            antigen_chains=",".join(complex_data.antigen_chain_ids),
            sabdab_id=str(record.get("SABDAB_ID", "")),
            ab_cluster=str(record.get("ab_cluster", "")),
            ag_cluster=str(record.get("agclusters", "")),
            ab_ag_cluster=str(record.get("ab_ag_cluster", "")),
            label=1,  # Positive cognate Ab-Ag complex
            representation_path=str(tensor_dest.relative_to(BASE_DIR) if tensor_dest.is_relative_to(BASE_DIR) else tensor_dest),
            representation_shape=str(rep.tensor_20x21.shape),
            intermolecular_contact_count=rep.intermolecular.contact_count,
            antibody_interface_count=rep.intermolecular.antibody_interface_count,
            cdr_interface_count=rep.intermolecular.cdr_interface_count,
            antigen_interface_count=rep.intermolecular.antigen_interface_count,
            antibody_intramolecular_contact_count=rep.antibody_intramolecular.total_contacts,
            antigen_intramolecular_contact_count=rep.antigen_intramolecular.total_contacts,
            representation_nonzero_count=rep.stats_20x21.nonzero_count,
            representation_sha256=rep.stats_20x21.sha256,
            processing_status="ACCEPTED",
            split=str(record.get("ab_ag_split", "unassigned")),
        )

        return sample, None

    def build_dataset(
        self,
        cached_only: bool = True,
        candidate_pdbs: Optional[Set[str]] = None,
        limit: Optional[int] = None,
    ) -> Tuple[List[DatasetSample], List[RejectedSample]]:
        """Execute complete dataset build and write manifests.

        Args:
            cached_only: If True, only attempts to process structures cached in structures_dir.
            candidate_pdbs: Optional filter set of PDB IDs.
            limit: Optional cap on processed samples.

        Returns:
            Tuple of (accepted_samples, rejected_samples).
        """
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        df = pd.read_csv(self.metadata_path)

        # Pre-filter by cached structures if requested
        if cached_only:
            cached_files = {p.stem.upper() for p in self.structures_dir.glob("*.cif")}
            cached_files.update({p.stem.upper() for p in self.structures_dir.glob("*.pdb")})
            mask = df["PDB_ID"].astype(str).apply(normalize_pdb_id).isin(cached_files)
            df = df[mask]

        if candidate_pdbs:
            norm_cands = {normalize_pdb_id(p) for p in candidate_pdbs}
            mask = df["PDB_ID"].astype(str).apply(normalize_pdb_id).isin(norm_cands)
            df = df[mask]

        if limit is not None and limit > 0:
            df = df.iloc[:limit]

        accepted: List[DatasetSample] = []
        rejected: List[RejectedSample] = []

        for _, row in df.iterrows():
            sample, rej = self.process_record(row)
            if sample:
                accepted.append(sample)
            elif rej:
                rejected.append(rej)

        # Write manifest CSVs
        self._write_manifests(accepted, rejected)

        return accepted, rejected

    def _write_manifests(
        self,
        accepted: List[DatasetSample],
        rejected: List[RejectedSample],
    ):
        """Write dataset_manifest.csv and rejected_samples.csv deterministically."""
        manifest_path = self.output_dir / "dataset_manifest.csv"
        rejected_path = self.output_dir / "rejected_samples.csv"

        if accepted:
            keys = list(accepted[0].to_dict().keys())
            with open(manifest_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for s in sorted(accepted, key=lambda x: x.sample_id):
                    writer.writerow(s.to_dict())
        else:
            manifest_path.touch()

        if rejected:
            keys = list(rejected[0].to_dict().keys())
            with open(rejected_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for r in sorted(rejected, key=lambda x: x.sample_id):
                    writer.writerow(r.to_dict())
        else:
            rejected_path.touch()
