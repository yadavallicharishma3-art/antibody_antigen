"""Machine learning pipeline: CNN architecture, dataset loading, training, evaluation, serialization, and inference."""

from .cnn import (
    build_antibody_antigen_cnn,
    get_architecture_summary,
)
from .data_loader import (
    DatasetBundle,
    load_dataset_split,
    load_all_splits,
    audit_splits_leakage,
    compute_representation_diversity,
    generate_mismatched_negatives,
)
from .train import (
    set_reproducibility_seeds,
    compute_training_class_weights,
    train_cnn_model,
)
from .evaluate import (
    compute_probability_diagnostics,
    evaluate_model_split,
    evaluate_all_splits,
    evaluate_unseen_structure,
)
from .serialization import (
    serialize_model_suite,
    reload_and_verify_model,
    AMINO_ACIDS_ORDER,
)
from .inference import (
    load_inference_model,
    predict_interaction,
    run_pipeline_for_pdb,
    validate_pdb_id,
    validate_inference_input,
    InferenceException,
    ModelLoadError,
    InvalidPdbIdError,
    StructureNotFoundError,
    InvalidComplexError,
    NoValidContactsError,
    DEFAULT_MODEL_PATH,
)

__all__ = [
    "build_antibody_antigen_cnn",
    "get_architecture_summary",
    "DatasetBundle",
    "load_dataset_split",
    "load_all_splits",
    "audit_splits_leakage",
    "compute_representation_diversity",
    "generate_mismatched_negatives",
    "set_reproducibility_seeds",
    "compute_training_class_weights",
    "train_cnn_model",
    "compute_probability_diagnostics",
    "evaluate_model_split",
    "evaluate_all_splits",
    "evaluate_unseen_structure",
    "serialize_model_suite",
    "reload_and_verify_model",
    "AMINO_ACIDS_ORDER",
    "load_inference_model",
    "predict_interaction",
    "run_pipeline_for_pdb",
    "validate_pdb_id",
    "validate_inference_input",
    "InferenceException",
    "ModelLoadError",
    "InvalidPdbIdError",
    "StructureNotFoundError",
    "InvalidComplexError",
    "NoValidContactsError",
    "DEFAULT_MODEL_PATH",
    "load_unified_cv_dataset",
    "generate_stratified_folds",
    "audit_fold_leakage",
    "run_stratified_cross_validation",
    "save_cross_validation_artifacts",
    "train_and_evaluate_single_fold",
    "DEFAULT_CV_DIR",
    "compute_general_ppi_representation",
    "build_test1_dataset",
    "train_test1_model",
    "evaluate_test1_partition",
    "run_test1_evaluation",
    "save_test1_artifacts",
    "DEFAULT_TEST1_DIR",
]

from .cross_validation import (
    load_unified_cv_dataset,
    generate_stratified_folds,
    audit_fold_leakage,
    run_stratified_cross_validation,
    save_cross_validation_artifacts,
    train_and_evaluate_single_fold,
    compute_metrics_from_predictions,
    DEFAULT_CV_DIR,
)
from .test1 import (
    compute_general_ppi_representation,
    build_test1_dataset,
    train_test1_model,
    evaluate_test1_partition,
    run_test1_evaluation,
    save_test1_artifacts,
    DEFAULT_TEST1_DIR,
)

