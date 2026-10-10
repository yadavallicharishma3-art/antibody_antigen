"""Command-line script to build, filter, split, and diagnose the antibody-antigen dataset."""

import sys
import argparse
from pathlib import Path

# Configure utf-8 stdout encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd
from backend.app.datasets import (
    DatasetBuilder,
    ClusterAwareSplitter,
    generate_dataset_report,
)
from backend.app.structure import (
    analyze_biological_entities,
    build_interaction_representation,
    resolve_structure_path,
)


def run_unseen_structure_validation(pdb_id: str = "1A3R") -> dict:
    """Validate an independent unseen structure not part of the Phase 3 benchmarks."""
    pdb_clean = pdb_id.strip().upper()
    was_cached = (BASE_DIR / "data" / "structures" / f"{pdb_clean}.cif").exists()

    try:
        struct_path = resolve_structure_path(pdb_clean, allow_download=True)
        comp = analyze_biological_entities(struct_path)
        rep = build_interaction_representation(comp)

        return {
            "pdb_id": pdb_clean,
            "was_cached": was_cached,
            "structure_path": str(struct_path),
            "heavy_chain": comp.antibody.heavy_chain_id,
            "light_chain": comp.antibody.light_chain_id,
            "antigen_chains": comp.antigen_chain_ids,
            "cdr_counts": {k: len(cdr.residues) for k, cdr in comp.antibody.cdrs.items()},
            "inter_contacts": rep.intermolecular.contact_count,
            "ab_interface": rep.intermolecular.antibody_interface_count,
            "cdr_interface": rep.intermolecular.cdr_interface_count,
            "ag_interface": rep.intermolecular.antigen_interface_count,
            "ab_intra": rep.antibody_intramolecular.total_contacts,
            "ag_intra": rep.antigen_intramolecular.total_contacts,
            "shape": str(rep.tensor_20x21.shape),
            "nonzero_count": rep.stats_20x21.nonzero_count,
            "sha256": rep.stats_20x21.sha256,
            "status": "SUCCESS",
        }
    except Exception as exc:
        return {
            "pdb_id": pdb_clean,
            "was_cached": was_cached,
            "status": "FAILED",
            "error": str(exc),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Build antibody-antigen dataset from SAbDab metadata and structural coordinates."
    )
    parser.add_argument("--metadata", type=str, default=None, help="Path to abag_split.csv")
    parser.add_argument("--structures", type=str, default=None, help="Path to data/structures/")
    parser.add_argument("--output", type=str, default=None, help="Path to data/processed/")
    parser.add_argument("--limit", type=int, default=None, help="Limit maximum samples to process")
    parser.add_argument("--allow-download", action="store_true", help="Download missing structures from RCSB")
    parser.add_argument("--all-metadata", action="store_true", help="Process all metadata rows (disables cached-only)")
    parser.add_argument("--val-fraction", type=float, default=0.20, help="Fraction of train pool for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic cluster split")
    parser.add_argument("--unseen-pdb", type=str, default="1A3R", help="PDB ID for unseen validation")
    args = parser.parse_args()

    meta_path = Path(args.metadata) if args.metadata else None
    struct_dir = Path(args.structures) if args.structures else None
    out_dir = Path(args.output) if args.output else None

    builder = DatasetBuilder(
        metadata_path=meta_path,
        structures_dir=struct_dir,
        output_dir=out_dir,
        allow_download=args.allow_download,
    )

    cached_only = not args.all_metadata
    print(f"Building dataset (cached_only={cached_only}, allow_download={args.allow_download}, limit={args.limit})...")
    accepted, rejected = builder.build_dataset(
        cached_only=cached_only,
        limit=args.limit,
    )

    # Cluster-aware splitting
    splitter = ClusterAwareSplitter(
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    train, val, test, metrics = splitter.split_samples(accepted)
    split_paths = splitter.export_splits(train, val, test, builder.output_dir)

    # Re-save manifest with updated split annotations
    builder._write_manifests(accepted, rejected)

    # Count source totals
    total_source_rows = 0
    total_source_pdbs = 0
    if builder.metadata_path.exists():
        src_df = pd.read_csv(builder.metadata_path, usecols=["PDB_ID"])
        total_source_rows = len(src_df)
        total_source_pdbs = src_df["PDB_ID"].nunique()

    # Diagnostic report
    report = generate_dataset_report(
        total_source_rows=total_source_rows,
        total_source_pdbs=total_source_pdbs,
        accepted_samples=accepted,
        rejected_samples=rejected,
        split_metrics=metrics,
    )
    print(report)

    # Unseen structure validation
    print("\n8. UNSEEN STRUCTURE VALIDATION")
    print("-" * 35)
    unseen_res = run_unseen_structure_validation(args.unseen_pdb)
    print(f"  PDB ID:               {unseen_res['pdb_id']}")
    print(f"  Locally cached:       {unseen_res['was_cached']}")
    print(f"  Status:               {unseen_res['status']}")
    if unseen_res['status'] == 'SUCCESS':
        print(f"  Heavy Chain:          {unseen_res['heavy_chain']}")
        print(f"  Light Chain:          {unseen_res['light_chain']}")
        print(f"  Antigen Chains:       {unseen_res['antigen_chains']}")
        print(f"  CDR Residue Counts:   {unseen_res['cdr_counts']}")
        print(f"  Inter contacts:       {unseen_res['inter_contacts']}")
        print(f"  Ab interface residues:{unseen_res['ab_interface']} (CDR: {unseen_res['cdr_interface']})")
        print(f"  Ag interface residues:{unseen_res['ag_interface']}")
        print(f"  Ab intra contacts:    {unseen_res['ab_intra']}")
        print(f"  Ag intra contacts:    {unseen_res['ag_intra']}")
        print(f"  Tensor shape:         {unseen_res['shape']}")
        print(f"  Nonzero count:        {unseen_res['nonzero_count']}")
        print(f"  SHA256 hash:          {unseen_res['sha256']}")
    else:
        print(f"  Error:                {unseen_res.get('error')}")


if __name__ == "__main__":
    main()
