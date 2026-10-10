"""Execute 10-Fold Stratified Cross-Validation for Antibody-Antigen Interaction CNN."""

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

from backend.app.ml import (
    load_unified_cv_dataset,
    run_stratified_cross_validation,
    save_cross_validation_artifacts,
    DEFAULT_CV_DIR,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run 10-fold stratified cross-validation on the Test-3 cognate-vs-mismatched classification task."
    )
    parser.add_argument("--epochs", type=int, default=60, help="Maximum epochs per fold (default: 60)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (default: 15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--n-splits", type=int, default=10, help="Number of folds (default: 10)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold (default: 0.5)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for CV artifacts (default: model/cv)")
    parser.add_argument("--verbose", type=int, default=0, help="Keras training verbosity (0=silent, 1=progress, default: 0)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else DEFAULT_CV_DIR

    print("=" * 72)
    print("PHASE 8: 10-FOLD STRATIFIED CROSS-VALIDATION PIPELINE")
    print("=" * 72)
    print(f"Task:              Zhang et al. 2024 Test-3-style cognate-vs-mismatched")
    print(f"Folds:             {args.n_splits}")
    print(f"Random seed:       {args.seed}")
    print(f"Training config:   epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}, patience={args.patience}")
    print(f"Threshold:         {args.threshold}")
    print(f"Output directory:  {out_dir}")
    print("-" * 72)

    # 1. Load authoritative Phase 5 dataset
    print("\n1. LOADING UNIFIED DATASET (Authoritative Phase 5 Corrected)...")
    dataset = load_unified_cv_dataset(seed=args.seed)
    print(f"   Total samples:     {len(dataset)} ({dataset.positive_count} Pos, {dataset.negative_count} Neg)")
    print(f"   Unique PDBs:       {len(dataset.unique_pdbs)}")
    print(f"   Tensor shape:      {dataset.X.shape}")

    # 2. Execute cross-validation
    print(f"\n2. EXECUTING {args.n_splits}-FOLD STRATIFIED CROSS-VALIDATION...")
    cv_results = run_stratified_cross_validation(
        bundle=dataset,
        n_splits=args.n_splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        seed=args.seed,
        threshold=args.threshold,
        verbose=args.verbose,
    )

    # 3. Print Leakage Audit
    audit = cv_results["leakage_audit"]
    print(f"\n3. CROSS-VALIDATION LEAKAGE & COVERAGE AUDIT:")
    print(f"   Zero sample overlap across train/val: {'VERIFIED' if audit['zero_sample_overlap'] else 'FAILED'}")
    print(f"   100% out-of-fold sample coverage:    {'VERIFIED' if audit['complete_coverage'] else 'FAILED'}")
    for fa in audit["fold_audits"]:
        print(f"   Fold {fa['fold']:2d}: Train={fa['train_size']:2d} (Pos={fa['train_pos']}, Neg={fa['train_neg']}) | "
              f"Val={fa['val_size']:2d} (Pos={fa['val_pos']}, Neg={fa['val_neg']}) | "
              f"PDB overlap={fa['pos_pdb_overlap']}")

    # 4. Print Macro Fold Statistics
    s = cv_results["macro_fold_statistics"]
    print(f"\n4. MACRO-AVERAGED FOLD METRICS across {args.n_splits} folds:")
    print(f"   Accuracy:  {s['accuracy']['mean'] * 100:6.2f}% ± {s['accuracy']['std'] * 100:5.2f}%   (min: {s['accuracy']['min'] * 100:.2f}%, max: {s['accuracy']['max'] * 100:.2f}%)")
    print(f"   ROC-AUC:   {s['roc_auc']['mean']:7.4f}  ± {s['roc_auc']['std']:6.4f}   (min: {s['roc_auc']['min']:.4f}, max: {s['roc_auc']['max']:.4f})")
    print(f"   Loss:      {s['loss']['mean']:7.4f}  ± {s['loss']['std']:6.4f}   (min: {s['loss']['min']:.4f}, max: {s['loss']['max']:.4f})")
    print(f"   Precision: {s['precision']['mean']:7.4f}  ± {s['precision']['std']:6.4f}   (min: {s['precision']['min']:.4f}, max: {s['precision']['max']:.4f})")
    print(f"   Recall:    {s['recall']['mean']:7.4f}  ± {s['recall']['std']:6.4f}   (min: {s['recall']['min']:.4f}, max: {s['recall']['max']:.4f})")
    print(f"   F1-Score:  {s['f1']['mean']:7.4f}  ± {s['f1']['std']:6.4f}   (min: {s['f1']['min']:.4f}, max: {s['f1']['max']:.4f})")

    # 5. Print Aggregate OOF Results
    agg = cv_results["aggregate_oof_metrics"]
    cm = agg["confusion_matrix"]
    print(f"\n5. AGGREGATE OUT-OF-FOLD (OOF) EVALUATION (N={len(dataset)}):")
    print(f"   OOF Accuracy:   {agg['accuracy'] * 100:.2f}% ({cm['true_positive'] + cm['true_negative']}/{len(dataset)})")
    print(f"   OOF ROC-AUC:    {agg['roc_auc']:.4f}")
    print(f"   OOF Precision:  {agg['precision']:.4f}")
    print(f"   OOF Recall:     {agg['recall']:.4f}")
    print(f"   OOF F1-Score:   {agg['f1']:.4f}")
    print(f"   OOF Loss (BCE): {agg['loss']:.4f}")
    print(f"   Confusion Matrix: TP={cm['true_positive']}, FP={cm['false_positive']}, TN={cm['true_negative']}, FN={cm['false_negative']}")

    # 6. Probability Diagnostics
    diag = cv_results["aggregate_oof_probability_diagnostics"]
    print(f"\n6. OUT-OF-FOLD PROBABILITY DIAGNOSTICS:")
    print(f"   Range:          [{diag['min']:.4f}, {diag['max']:.4f}]")
    print(f"   Mean:           {diag['mean']:.4f} (median: {diag['median']:.4f})")
    print(f"   Std Dev:        {diag['std']:.4f}")
    print(f"   Unique values:  {diag['unique_count']}")
    print(f"   In [0.45, 0.55]: {diag['fraction_near_05'] * 100:.1f}%")
    print(f"   Assessment:     {diag['collapse_assessment']}")

    # 7. Serialize Artifacts
    print(f"\n7. SERIALIZING PHASE 8 CROSS-VALIDATION ARTIFACTS TO {out_dir}...")
    saved_paths = save_cross_validation_artifacts(cv_results, output_dir=out_dir)
    for name, p in saved_paths.items():
        print(f"   Saved {name:15s}: {p}")

    print("\n" + "=" * 72)
    print("PHASE 8 CROSS-VALIDATION COMPLETED SUCCESSFULLY")
    print("=" * 72)


if __name__ == "__main__":
    main()
