"""Evaluate a serialized antibody-antigen CNN model on test split and unseen structures."""

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
import tensorflow as tf

from backend.app.ml import (
    load_all_splits,
    evaluate_model_split,
)
from backend.app.structure import (
    analyze_biological_entities,
    build_interaction_representation,
    resolve_structure_path,
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate saved antibody-antigen CNN model."
    )
    parser.add_argument("--model-path", type=str, default=None, help="Path to saved .keras model file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification decision threshold")
    parser.add_argument("--unseen-pdb", type=str, default="1A3R", help="PDB ID for unseen structure evaluation")
    args = parser.parse_args()

    model_path = Path(args.model_path) if args.model_path else (BASE_DIR / "model" / "antibody_antigen_cnn.keras")
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    print("=" * 65)
    print("PHASE 5: MODEL EVALUATION & PROBABILITY AUDIT")
    print("=" * 65)
    print(f"Loading serialized model from: {model_path} ...")
    model = tf.keras.models.load_model(model_path)

    # Load splits
    _, _, test_bundle, _ = load_all_splits()
    print(f"Loaded held-out test split: {len(test_bundle)} samples from {len(test_bundle.unique_pdbs)} unique PDBs.")

    # Evaluate on test split
    metrics = evaluate_model_split(model, test_bundle, threshold=args.threshold)
    auc_str = f"{metrics['roc_auc']:.4f}" if not np.isnan(metrics["roc_auc"]) else f"N/A ({metrics['auc_note']})"

    print(f"\n1. TEST SPLIT METRICS (Threshold = {args.threshold}):")
    print(f"   Samples:        {metrics['sample_count']}")
    print(f"   Loss:           {metrics['loss']:.4f}")
    print(f"   Accuracy:       {metrics['accuracy'] * 100:.2f}%")
    print(f"   ROC-AUC:        {auc_str}")
    print(f"   Precision:      {metrics['precision']:.4f}")
    print(f"   Recall:         {metrics['recall']:.4f}")
    print(f"   F1-Score:       {metrics['f1']:.4f}")
    print(f"   Confusion:      TP={metrics['confusion_matrix']['true_positive']}, FP={metrics['confusion_matrix']['false_positive']}, "
          f"TN={metrics['confusion_matrix']['true_negative']}, FN={metrics['confusion_matrix']['false_negative']}")

    print(f"\n2. TEST PROBABILITY DISTRIBUTION STATISTICS:")
    p = metrics["probability_diagnostics"]
    print(f"   Min probability:     {p['min']:.4f}")
    print(f"   Max probability:     {p['max']:.4f}")
    print(f"   Mean probability:    {p['mean']:.4f}")
    print(f"   Median probability:  {p['median']:.4f}")
    print(f"   Standard deviation:  {p['std']:.4f}")
    print(f"   Unique values (4d):  {p['unique_count']}")
    print(f"   Fraction near 0.5:   {p['fraction_near_05'] * 100:.1f}%")
    print(f"   Collapse status:     {p['collapse_assessment']}")

    print(f"\n3. SAMPLE-LEVEL PREDICTIONS:")
    for sp in metrics["sample_predictions"]:
        status = "CORRECT" if sp["correct"] else "INCORRECT"
        print(f"   {sp['sample_id']:25s} | True={sp['true_label']} | Pred={sp['pred_label']} (P={sp['pred_probability']:.4f}) | {status}")

    # Unseen structure
    if args.unseen_pdb:
        print(f"\n4. UNSEEN STRUCTURE EVALUATION ({args.unseen_pdb}):")
        try:
            struct_path = resolve_structure_path(args.unseen_pdb.strip().upper(), allow_download=False)
            comp = analyze_biological_entities(struct_path)
            rep = build_interaction_representation(comp)
            tensor_input = np.expand_dims(rep.tensor_20x21.astype(np.float32), axis=(0, -1))
            prob = float(model.predict(tensor_input, verbose=0)[0, 0])
            pred_cls = int(prob >= args.threshold)
            print(f"   PDB ID:          {args.unseen_pdb.upper()}")
            print(f"   Probability:     {prob:.4f}")
            print(f"   Class:           {pred_cls}")
            print(f"   Contacts:        {rep.intermolecular.contact_count}")
            print(f"   Nonzero cells:   {rep.stats_20x21.nonzero_count}")
        except Exception as exc:
            print(f"   Failed to evaluate {args.unseen_pdb}: {exc}")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
