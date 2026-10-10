"""Execute Phase 9 Test 1 Scientific Evaluation (Antibody-Antigen vs General PPI)."""

import sys
import argparse
from pathlib import Path

# Configure utf-8 stdout encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.app.ml import (
    run_test1_evaluation,
    save_test1_artifacts,
    DEFAULT_TEST1_DIR,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run Test 1 evaluation: Antibody-Antigen vs General Protein-Protein Interactions."
    )
    parser.add_argument("--epochs", type=int, default=60, help="Max training epochs (default: 60)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (default: 15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold (default: 0.5)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: model/test1)")
    parser.add_argument("--verbose", type=int, default=1, help="Keras training verbosity (default: 1)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else DEFAULT_TEST1_DIR

    print("=" * 72)
    print("PHASE 9: TEST 1 SCIENTIFIC EVALUATION PIPELINE")
    print("=" * 72)
    print(f"Task:             Antibody-Antigen vs General PPI (Zhang et al. 2024 Test 1)")
    print(f"Random seed:      {args.seed}")
    print(f"Threshold:        {args.threshold}")
    print(f"Output directory: {out_dir}")
    print("-" * 72)

    results = run_test1_evaluation(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        seed=args.seed,
        threshold=args.threshold,
        verbose=args.verbose,
    )

    ds = results["dataset_summary"]
    print("\n1. TEST 1 DATASET SUMMARY:")
    print(f"   Total samples:     {ds['total_samples']} (Pos: {ds['positive_count']}, Neg: {ds['negative_count']})")
    print(f"   Train split:       {ds['train_samples']} samples (Pos: {ds['train_pos']}, Neg: {ds['train_neg']})")
    print(f"   Validation split:  {ds['val_samples']} samples (Pos: {ds['val_pos']}, Neg: {ds['val_neg']})")
    print(f"   Test split:        {ds['test_samples']} samples (Pos: {ds['test_pos']}, Neg: {ds['test_neg']})")
    print(f"   Unique Pos PDBs:   {ds['unique_positive_pdbs']}")
    print(f"   Unique Neg PDBs:   {ds['unique_negative_pdbs']}")
    print(f"   Zero sample leakage: {'VERIFIED' if ds['zero_sample_overlap'] else 'FAILED'}")

    te = results["evaluation"]["test"]
    cm = te["confusion_matrix"]
    print("\n2. HELD-OUT TEST SPLIT PERFORMANCE:")
    print(f"   Accuracy:          {te['accuracy'] * 100:.2f}% ({cm['true_positive'] + cm['true_negative']}/{te['sample_count']})")
    print(f"   ROC-AUC:           {te['roc_auc']:.4f}")
    print(f"   Precision:         {te['precision']:.4f}")
    print(f"   Recall:            {te['recall']:.4f}")
    print(f"   F1-Score:          {te['f1']:.4f}")
    print(f"   Loss (BCE):        {te['loss']:.4f}")
    print(f"   Confusion Matrix:  TP={cm['true_positive']}, FP={cm['false_positive']}, TN={cm['true_negative']}, FN={cm['false_negative']}")

    pd_ = te["probability_diagnostics"]
    print("\n3. TEST PROBABILITY DIAGNOSTICS:")
    print(f"   Range:             [{pd_['min']:.4f}, {pd_['max']:.4f}]")
    print(f"   Mean:              {pd_['mean']:.4f} (median: {pd_['median']:.4f})")
    print(f"   Std Dev:           {pd_['std']:.4f}")
    print(f"   Unique values:     {pd_['unique_count']}")
    print(f"   Collapse status:   {pd_['collapse_assessment']}")

    print("\n4. SAMPLE PREDICTIONS ON HELD-OUT TEST SPLIT:")
    for sp in te["sample_predictions"]:
        status = "CORRECT" if sp["correct"] else "INCORRECT"
        print(f"   {sp['sample_id']:25s} | Type={sp['class_type']:16s} | True={sp['true_label']} | Pred={sp['pred_label']} (P={sp['pred_probability']:.4f}) | {status}")

    print(f"\n5. SAVING TEST 1 ARTIFACTS TO {out_dir}...")
    saved = save_test1_artifacts(results, output_dir=out_dir)
    for k, v in saved.items():
        print(f"   Saved {k:18s}: {v}")

    print("\n" + "=" * 72)
    print("PHASE 9 TEST 1 EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 72)


if __name__ == "__main__":
    main()
