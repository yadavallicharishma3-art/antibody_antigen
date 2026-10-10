"""Train, evaluate, and serialize the antibody-antigen CNN model with zero data leakage."""

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
    build_antibody_antigen_cnn,
    get_architecture_summary,
    load_all_splits,
    compute_representation_diversity,
    generate_mismatched_negatives,
    train_cnn_model,
    evaluate_all_splits,
    evaluate_model_split,
    evaluate_unseen_structure,
    serialize_model_suite,
    reload_and_verify_model,
    set_reproducibility_seeds,
)


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate antibody-antigen CNN following Zhang et al. 2024."
    )
    parser.add_argument("--epochs", type=int, default=60, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Adam optimizer")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save serialized model")
    parser.add_argument("--unseen-pdb", type=str, default="1A14", help="PDB ID for unseen structure evaluation (default: 1A14)")
    parser.add_argument("--mismatched-negatives", dest="mismatched_negatives", action="store_true", default=True, help="Generate balanced non-cognate negative pairs (Test 3 paradigm, default: True)")
    parser.add_argument("--no-mismatched-negatives", dest="mismatched_negatives", action="store_false", help="Disable non-cognate negative pair generation")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate (default 0.0)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else (BASE_DIR / "model")

    print("=" * 70)
    print("PHASE 5: CNN IMPLEMENTATION, TRAINING & EVALUATION PIPELINE")
    print("=" * 70)

    # 1. Reproducibility setup
    seed_info = set_reproducibility_seeds(args.seed)
    print(f"\n1. REPRODUCIBILITY SEEDS CONFIGURED:")
    print(f"   Python seed:      {seed_info['python_seed']}")
    print(f"   NumPy seed:       {seed_info['numpy_seed']}")
    print(f"   TensorFlow seed:  {seed_info['tensorflow_seed']}")

    # 2. Load Phase 4 splits
    print(f"\n2. LOADING PHASE 4 DATASET SPLITS (Source of Truth)...")
    train_bundle, val_bundle, test_bundle, leakage_audit = load_all_splits()

    if args.mismatched_negatives:
        print("   [INFO] Generating within-split non-cognate negative pairs (Test 3)...")
        train_bundle = generate_mismatched_negatives(train_bundle, seed=args.seed)
        val_bundle = generate_mismatched_negatives(val_bundle, seed=args.seed)
        test_bundle = generate_mismatched_negatives(test_bundle, seed=args.seed)

    print(f"   Train samples:      {len(train_bundle)} (Pos: {train_bundle.positive_count}, Neg: {train_bundle.negative_count})")
    print(f"   Validation samples: {len(val_bundle)} (Pos: {val_bundle.positive_count}, Neg: {val_bundle.negative_count})")
    print(f"   Test samples:       {len(test_bundle)} (Pos: {test_bundle.positive_count}, Neg: {test_bundle.negative_count})")
    print(f"   Total samples:      {len(train_bundle) + len(val_bundle) + len(test_bundle)}")

    # 3. Leakage audit
    print(f"\n3. STRUCTURE-LEVEL LEAKAGE AUDIT:")
    print(f"   Train unique PDBs:        {len(train_bundle.unique_pdbs)}: {sorted(list(train_bundle.unique_pdbs))}")
    print(f"   Val unique PDBs:          {len(val_bundle.unique_pdbs)}: {sorted(list(val_bundle.unique_pdbs))}")
    print(f"   Test unique PDBs:         {len(test_bundle.unique_pdbs)}: {sorted(list(test_bundle.unique_pdbs))}")
    print(f"   Overlap (Train ∩ Val):    {len(leakage_audit['pdb_overlap_train_val'])} {leakage_audit['pdb_overlap_train_val']}")
    print(f"   Overlap (Train ∩ Test):   {len(leakage_audit['pdb_overlap_train_test'])} {leakage_audit['pdb_overlap_train_test']}")
    print(f"   Overlap (Val ∩ Test):     {len(leakage_audit['pdb_overlap_val_test'])} {leakage_audit['pdb_overlap_val_test']}")
    print(f"   Cluster overlap (Tr/Te):  {len(leakage_audit['ab_ag_cluster_overlap_train_test'])}")
    print(f"   Leakage-free status:      {'VERIFIED (0 overlap)' if leakage_audit['is_leakage_free'] else 'LEAKAGE DETECTED'}")

    assert leakage_audit["is_leakage_free"], "Fatal: Data leakage detected across splits!"

    # 4. Representation diversity check
    print(f"\n4. REPRESENTATION DIVERSITY DIAGNOSTICS:")
    train_div = compute_representation_diversity(train_bundle)
    print(f"   Train unique hashes:      {train_div['unique_representation_hashes']} / {train_div['total_samples']}")
    print(f"   Nonzero cells per tensor: min={train_div['nonzero_cells_min']}, max={train_div['nonzero_cells_max']}, mean={train_div['nonzero_cells_mean']:.2f}")
    print(f"   Matrix value range:       [{train_div['matrix_val_min']:.4f}, {train_div['matrix_val_max']:.4f}], mean={train_div['matrix_val_mean']:.4f}")
    print(f"   Mean pairwise distance:   {train_div['mean_pairwise_frobenius_distance']:.4f}")
    print(f"   Collapsed status:         {'COLLAPSED' if train_div['is_representation_collapsed'] else 'HEALTHY (Diverse representations)'}")

    # 5. Architecture verification
    print(f"\n5. CNN ARCHITECTURE VERIFICATION (Zhang et al. 2024):")
    model = build_antibody_antigen_cnn(
        input_shape=(20, 21, 1),
        learning_rate=args.lr,
        dropout_rate=args.dropout,
    )
    arch = get_architecture_summary(model)
    print(f"   Input shape:              {arch['input_shape']}")
    print(f"   Conv stages:              3 stages (32 filters, 3x3 kernels, same padding, relu)")
    print(f"   Pooling stages:           3 MaxPooling2D (2x2, same padding)")
    print(f"   Flatten bottleneck:       {arch['flatten_dim']} dimensions (exact match to paper)")
    print(f"   Dense layers:             Dense(512, relu) -> Dense(64, relu) -> Dense(1, sigmoid)")
    print(f"   Total trainable params:   {arch['trainable_parameters']:,}")

    # 6. Training
    print(f"\n6. MODEL TRAINING (Train split only, early stopping on validation):")
    print(f"   Requested epochs: {args.epochs}, Batch size: {args.batch_size}, Learning rate: {args.lr}")
    trained_model, train_meta = train_cnn_model(
        train_bundle=train_bundle,
        val_bundle=val_bundle,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        seed=args.seed,
        dropout_rate=args.dropout,
        verbose=1,
    )
    print(f"   Training completed:       {train_meta['epochs_completed']} epochs (best epoch: {train_meta['best_epoch']})")
    print(f"   Class weight rationale:   {train_meta['class_weight_rationale']}")

    # 7. Evaluation across splits
    print(f"\n7. EVALUATING MODEL ON HELD-OUT SPLITS:")
    eval_results = evaluate_all_splits(trained_model, train_bundle, val_bundle, test_bundle)

    for split_key in ["train", "validation", "test"]:
        m = eval_results[split_key]
        auc_str = f"{m['roc_auc']:.4f}" if not np.isnan(m["roc_auc"]) else f"N/A ({m['auc_note']})"
        print(f"\n   [{split_key.upper()} SPLIT - {m['sample_count']} samples]")
        print(f"     Loss:      {m['loss']:.4f}")
        print(f"     Accuracy:  {m['accuracy'] * 100:.2f}%")
        print(f"     ROC-AUC:   {auc_str}")
        print(f"     Precision: {m['precision']:.4f}")
        print(f"     Recall:    {m['recall']:.4f}")
        print(f"     F1-Score:  {m['f1']:.4f}")
        print(f"     Confusion: TP={m['confusion_matrix']['true_positive']}, FP={m['confusion_matrix']['false_positive']}, "
              f"TN={m['confusion_matrix']['true_negative']}, FN={m['confusion_matrix']['false_negative']}")

    # 8. Test set probability distribution check
    print(f"\n8. TEST SET PROBABILITY DISTRIBUTION CHECK (Critical Anti-Collapse Audit):")
    test_probs = eval_results["test"]["probability_diagnostics"]
    print(f"   Probability min:          {test_probs['min']:.4f}")
    print(f"   Probability max:          {test_probs['max']:.4f}")
    print(f"   Probability mean:         {test_probs['mean']:.4f}")
    print(f"   Probability median:       {test_probs['median']:.4f}")
    print(f"   Probability std:          {test_probs['std']:.4f}")
    print(f"   Unique values (4 dec):    {test_probs['unique_count']}")
    print(f"   Fraction near 0.5 [±0.05]:{test_probs['fraction_near_05'] * 100:.1f}%")
    print(f"   Collapse diagnostic:      {test_probs['collapse_assessment']}")

    # 9. Unseen structure evaluation
    print(f"\n9. INDEPENDENT UNSEEN STRUCTURE VALIDATION ({args.unseen_pdb}):")
    unseen_res = evaluate_unseen_structure(trained_model, pdb_id=args.unseen_pdb)
    eval_results["unseen_structure"] = unseen_res
    print(f"   PDB ID:                   {unseen_res.get('pdb_id')}")
    print(f"   Status:                   {unseen_res.get('status')}")
    if unseen_res.get("status") == "SUCCESS":
        print(f"   Predicted Probability:    {unseen_res['prediction_probability']:.4f}")
        print(f"   Predicted Class:          {unseen_res['prediction_label']} (Threshold: {unseen_res['threshold']})")
        print(f"   Intermolecular contacts:  {unseen_res['intermolecular_contacts']}")
        print(f"   CDR interface residues:   {unseen_res['cdr_interface_residues']}")
        print(f"   Antigen interface res:    {unseen_res['antigen_interface_residues']}")
    else:
        print(f"   Error:                    {unseen_res.get('error')}")

    # 10. Model serialization
    print(f"\n10. MODEL SERIALIZATION:")
    saved_paths = serialize_model_suite(
        model=trained_model,
        training_meta=train_meta,
        evaluation_results=eval_results,
        train_bundle=train_bundle,
        val_bundle=val_bundle,
        test_bundle=test_bundle,
        output_dir=out_dir,
    )
    for k, p in saved_paths.items():
        print(f"   Saved {k:10s}: {p}")

    # 11. Model reload verification
    print(f"\n11. MODEL RELOAD VERIFICATION TEST:")
    orig_test_probs = np.array([p["pred_probability"] for p in eval_results["test"]["sample_predictions"]])
    reload_res = reload_and_verify_model(
        saved_model_path=saved_paths["model"],
        test_bundle=test_bundle,
        in_memory_predictions=orig_test_probs,
        tolerance=1e-5,
    )
    print(f"   Reloaded model path:      {reload_res['saved_model_path']}")
    print(f"   Max absolute difference:  {reload_res['max_absolute_difference']:.8e}")
    print(f"   Mean absolute difference: {reload_res['mean_absolute_difference']:.8e}")
    print(f"   Tolerance threshold:      {reload_res['tolerance']}")
    print(f"   Verification status:      {reload_res['verification_status']}")

    assert reload_res["within_tolerance"], "Fatal: Reloaded model predictions differ from in-memory model!"

    print("\n" + "=" * 70)
    print("PHASE 5 TRAINING, EVALUATION & SERIALIZATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
