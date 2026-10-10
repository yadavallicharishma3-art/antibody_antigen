"""Diagnostic command-line tool for analyzing biological entities and matrix representations

in antibody-antigen complexes.
"""

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

import numpy as np

from backend.app.structure import (
    analyze_biological_entities,
    generate_chain_inventory,
    ComplexData,
    build_interaction_representation,
    export_representation,
    compute_matrix_statistics,
    RepresentationResult,
)


def format_structure_diagnostic(complex_data: ComplexData, verbose: bool = False) -> str:
    """Format structured diagnostic report according to Phase 2 specifications."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"STRUCTURE: {complex_data.pdb_id}")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Source:")
    lines.append(f"  {complex_data.source_path}")
    lines.append("")
    lines.append("Models:")
    lines.append(f"  {complex_data.models_count}")
    lines.append("")

    # 1. Chain Inventory
    inventory = generate_chain_inventory(complex_data)
    lines.append("ALL CHAINS")
    for cid, stats in sorted(inventory.items()):
        ctype = complex_data.chains[cid].chain_type
        lines.append(
            f"  Chain {cid} ({ctype}): {stats['standard_residues']} standard residues "
            f"({stats['total_residues']} total residues, {stats['total_atoms']} atoms)"
        )
    lines.append("")

    # 2. Antibody
    lines.append("ANTIBODY")
    ab = complex_data.antibody
    lines.append(f"  Heavy chain: {ab.heavy_chain_id or 'None'}")
    lines.append(f"  Light chain: {ab.light_chain_id or 'None'}")
    lines.append(f"  Evidence: {ab.evidence or 'None'}")
    lines.append("")

    # 3. Antigen
    lines.append("ANTIGEN")
    ag_str = ", ".join(complex_data.antigen_chain_ids) if complex_data.antigen_chain_ids else "None"
    lines.append(f"  Chains: {ag_str}")
    lines.append(f"  Evidence: {complex_data.antigen_evidence or 'None'}")
    lines.append("")

    # 4. CDRs
    lines.append("CDRs")
    for cdr_name in ["H1", "H2", "H3", "L1", "L2", "L3"]:
        cdr = ab.cdrs.get(cdr_name)
        if cdr:
            lines.append(f"  {cdr_name}: {cdr.sequence} (expected: {cdr.expected_sequence or 'N/A'})")
        else:
            lines.append(f"  {cdr_name}: Not identified")
    lines.append("")

    # 5. CDR Residue Counts
    lines.append("CDR RESIDUE COUNTS")
    for cdr_name in ["H1", "H2", "H3", "L1", "L2", "L3"]:
        cdr = ab.cdrs.get(cdr_name)
        count = cdr.count if cdr else 0
        lines.append(f"  {cdr_name}: {count}")
    lines.append("")

    # 6. Mapping Status
    lines.append("MAPPING")
    for cdr_name in ["H1", "H2", "H3", "L1", "L2", "L3"]:
        cdr = ab.cdrs.get(cdr_name)
        if cdr and cdr.mapping_status == "MATCH":
            lines.append(f"  {cdr_name}: PASS ({cdr.notes})")
        elif cdr:
            lines.append(f"  {cdr_name}: {cdr.mapping_status} ({cdr.notes})")
        else:
            lines.append(f"  {cdr_name}: FAIL (Missing metadata or alignment)")
    lines.append("")

    # 7. Verbose Residue details if requested
    if verbose:
        lines.append("DETAILED CDR RESIDUES")
        for cdr_name in ["H1", "H2", "H3", "L1", "L2", "L3"]:
            cdr = ab.cdrs.get(cdr_name)
            if cdr:
                res_ids = [r.residue_id for r in cdr.residues]
                lines.append(f"  {cdr_name} residues: {', '.join(res_ids)}")
        lines.append("")

    # 8. Warnings
    lines.append("WARNINGS")
    if complex_data.warnings:
        for w in complex_data.warnings:
            lines.append(f"  - {w}")
    else:
        lines.append("  None")
    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def format_matrix_diagnostic(rep: RepresentationResult, verbose: bool = False, max_contacts: int = 10) -> str:
    """Format structured matrix diagnostic report according to Phase 3 Section 16 & 19 specifications."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"MATRIX DIAGNOSTIC: {rep.pdb_id}")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Shape:")
    lines.append(f"  20 x 20 (canonical) / 20 x 21 (CNN input tensor)")
    lines.append("")
    lines.append("Intermolecular contacts:")
    lines.append(f"  {rep.intermolecular.contact_count}")
    lines.append("")
    lines.append("Antibody interface residues:")
    lines.append(f"  {rep.intermolecular.antibody_interface_count}")
    lines.append("")
    lines.append("Antibody CDR interface residues:")
    lines.append(f"  {rep.intermolecular.cdr_interface_count}")
    lines.append("")
    lines.append("Antigen interface residues:")
    lines.append(f"  {rep.intermolecular.antigen_interface_count}")
    lines.append("")
    lines.append("Antibody intramolecular contacts:")
    lines.append(f"  {rep.antibody_intramolecular.total_contacts}")
    lines.append("")
    lines.append("Antigen intramolecular contacts:")
    lines.append(f"  {rep.antigen_intramolecular.total_contacts}")
    lines.append("")

    # Build raw 20x20 matrix for raw matrix statistics
    raw_20x20 = np.zeros((20, 20), dtype=np.float32)
    for r in range(20):
        for c in range(r, 20):
            raw_20x20[r, c] = rep.raw_ag_triangle[r, c] if not np.isnan(rep.raw_ag_triangle[r, c]) else 0.0
            raw_20x20[c, r] = rep.raw_ab_triangle[r, c] if not np.isnan(rep.raw_ab_triangle[r, c]) else 0.0

    raw_stats = compute_matrix_statistics(raw_20x20)
    norm_stats = rep.stats_20x20
    tensor_stats = rep.stats_20x21

    lines.append("Raw matrix:")
    lines.append(f"  min: {raw_stats.min_val:.6f}")
    lines.append(f"  max: {raw_stats.max_val:.6f}")
    lines.append(f"  mean: {raw_stats.mean_val:.6f}")
    lines.append(f"  std: {raw_stats.std_val:.6f}")
    lines.append(f"  nonzero: {raw_stats.nonzero_count}")
    lines.append(f"  sum: {raw_stats.sum_val:.6f}")
    lines.append("")
    lines.append("Normalized matrix:")
    lines.append(f"  min: {norm_stats.min_val:.6f}")
    lines.append(f"  max: {norm_stats.max_val:.6f}")
    lines.append(f"  mean: {norm_stats.mean_val:.6f}")
    lines.append(f"  std: {norm_stats.std_val:.6f}")
    lines.append(f"  nonzero: {norm_stats.nonzero_count}")
    lines.append(f"  sum: {norm_stats.sum_val:.6f}")
    lines.append("")
    lines.append("SHA256:")
    lines.append(f"  20x20: {norm_stats.sha256}")
    lines.append(f"  20x21: {tensor_stats.sha256}")
    lines.append("")

    if verbose:
        lines.append("-" * 40)
        lines.append("INTERMOLECULAR CONTACTS (sample)")
        lines.append("-" * 40)
        for contact in rep.intermolecular.contacts[:max_contacts]:
            lines.append(f"  {contact}")
        if len(rep.intermolecular.contacts) > max_contacts:
            lines.append(f"  ... ({len(rep.intermolecular.contacts) - max_contacts} more contacts)")
        lines.append("")

        lines.append("-" * 40)
        lines.append("ANTIGEN INTRAMOLECULAR CONTACTS (sample)")
        lines.append("-" * 40)
        for r1, r2 in rep.antigen_intramolecular.residue_pairs[:max_contacts]:
            lines.append(f"  {r1.residue_id} {r1.residue_name} ↔ {r2.residue_id} {r2.residue_name}")
        if len(rep.antigen_intramolecular.residue_pairs) > max_contacts:
            lines.append(f"  ... ({len(rep.antigen_intramolecular.residue_pairs) - max_contacts} more contacts)")
        lines.append("")

        lines.append("-" * 40)
        lines.append("ANTIBODY CDR INTRAMOLECULAR CONTACTS (sample)")
        lines.append("-" * 40)
        for r1, r2 in rep.antibody_intramolecular.residue_pairs[:max_contacts]:
            lines.append(f"  {r1.residue_id} {r1.residue_name} ↔ {r2.residue_id} {r2.residue_name}")
        if len(rep.antibody_intramolecular.residue_pairs) > max_contacts:
            lines.append(f"  ... ({len(rep.antibody_intramolecular.residue_pairs) - max_contacts} more contacts)")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose structure parsing, biological entities, and 5 Å interaction matrices."
    )
    parser.add_argument("pdb_id", help="PDB ID or structure filepath (e.g. 1EJO)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed contacts and CDR residues")
    parser.add_argument("--export", action="store_true", help="Export .npy and .csv matrix files")
    parser.add_argument("--phase", choices=["structure", "matrix", "all"], default="all",
                        help="Which diagnostic report to print (default: all)")
    args = parser.parse_args()

    complex_data = analyze_biological_entities(args.pdb_id)

    if args.phase in ["structure", "all"]:
        struct_report = format_structure_diagnostic(complex_data, verbose=args.verbose)
        print(struct_report)

    if args.phase in ["matrix", "all"]:
        rep = build_interaction_representation(complex_data)
        matrix_report = format_matrix_diagnostic(rep, verbose=args.verbose)
        print(matrix_report)

        if args.export:
            exported = export_representation(rep)
            print("\nExported matrix files:")
            for key, path in exported.items():
                print(f"  {key}: {path}")


if __name__ == "__main__":
    main()

