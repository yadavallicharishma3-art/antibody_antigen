# Antibody–Antigen Interaction Classification

A beginner-oriented FastAPI and TensorFlow project that classifies a protein complex as an antibody–antigen interaction or a general protein–protein interaction (PPI). Training uses the supplied datasets. For an interactive prediction, the app accepts a PDB ID or a local PDB/mmCIF file and detects contacting protein chains automatically.

## Folder layout

```text
Antibody_Antigen/
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── backend/
│   ├── app.py
│   ├── ml_pipeline.py
│   ├── structure_processing.py
│   └── dataset.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── requirements.txt
└── README.md
```

The raw archives remain at their provided locations by default; they are not copied into this project. `data/raw/` is available if you want to keep local copies. The scripts accept environment variables or command-line paths so the project can be moved to another computer.

## Supplied dataset sources

- Positive class: `splits_final.zip`, using `splits_final/abag_split.csv` and the paired per-complex CIF structures. The records provide antibody heavy/light chain IDs and antigen chain IDs. Each CSV row is an antibody–antigen example; `ab_ag_split` is a source split label, not the class label.
- Negative class: `pdb_4960list.txt`, matched by structure ID to the CIF/PDB files in `ppi_structures.zip`. The control-list provenance and the supplied study identify these as general protein–protein complexes.

The inspected supplied files contain 15,641 positive metadata rows and 4,144 negative-list rows. The negative list is smaller than the 4,960 complexes described in the paper. IDs `2j5t` and `2j5d` have no exact basename match in the provided PPI archive. Dataset processing does not fetch replacement files. `dataset.py --inspect` reports the counts and match status again before feature processing. PDB-ID prediction separately retrieves that user-requested structure from the RCSB file service.

The positive CSV source split has 111 PDB identifiers appearing in both its `train` and `test` partitions. To prevent leakage, the processing pipeline records the original split field but creates new reproducible, group-aware train/validation/test splits using the structure/PDB identifier. Related rows from one structure therefore stay in one partition.

## Architecture and workflow

1. `dataset.py` reads metadata directly from the ZIP archive, joins each row to a structure by its identifier, and writes a record-by-record manifest.
2. `structure_processing.py` parses PDB/mmCIF files, checks chains, residues, coordinates, and contacts, then creates a matrix only if every required check passes. Interactive prediction scans the structure for contacting protein-chain pairs.
3. `dataset.py` saves accepted matrices and labels, records every rejection reason, and creates group-aware data partitions.
4. `ml_pipeline.py` runs group-aware stratified cross-validation on the training portion with a fresh CNN per fold, then trains with a separate validation set and evaluates once on the untouched test set.
5. `app.py` exposes FastAPI PDB-ID lookup or file upload, automatic chain-pair detection, validation, and prediction. The dashboard shows the prediction and interaction fingerprint without training/evaluation panels.

## Feature definition

The feature design follows the scientific distinction described in the supplied paper and the project specification. A cross-partner heavy-atom distance of at most 5 Å defines interface residues. Then, for each partner separately, the pipeline counts side-chain heavy-atom contacts between distinct interface residues within that partner. The positive partner A is the antibody (heavy and light chains together); positive partner B is the antigen chain set. For negative examples, partner A and B are the two protein partners listed in the PPI metadata.

This is an intrapartner interface feature representation. The cross-partner contacts select the interface; they are not misrepresented as two separate cross-chain amino-acid matrices. For positive examples, this implementation uses all selected antibody interface residues and does not restrict them to CDR loops. A chain ID is matched case-insensitively only if that gives one unique chain in the structure. Side-chain contacts are counted once per residue pair. Hydrogens/deuteriums and non-standard residues are excluded; Biopython's selected alternate conformer is used.

The fixed amino-acid order is `ACDEFGHIKLMNPQRSTVWY`. For an amino-acid pair with indices `i` and `j`, the antigen/general partner's unordered count is stored at `[min(i,j), max(i,j)]` in the upper triangle. The antibody/first partner's count is stored at `[max(i,j), min(i,j)]` in the lower triangle. The diagonal is zero. Since the diagonal must remain zero, same-amino-acid contacts are tracked in metadata and omitted from the matrix. For each partner independently, every represented off-diagonal cell is divided by that partner's represented total; each triangle sums to 1. A sample is rejected if either partner has no represented contact counts.

## Setup

Use Python 3.10–3.12 in a virtual environment. TensorFlow wheels depend on operating system and Python version; use a supported Python runtime if installation reports that no wheel is available.

```powershell
cd Antibody_Antigen
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Default data paths point to the supplied files on the computer used to prepare this project. On another machine, set:

```powershell
$env:PPI_LIST_PATH = "D:\data\pdb_4960list.txt"
$env:PPI_STRUCTURES_ZIP = "D:\data\ppi_structures.zip"
$env:SPLITS_FINAL_ZIP = "D:\data\splits_final.zip"
```

Alternatively, pass `--ppi-list`, `--ppi-zip`, and `--abag-zip` to `dataset.py`.

## Commands

First inspect record counts and matching without parsing all structures:

```powershell
python -m backend.dataset --inspect
```

Process every supplied record with a default 5 Å contact threshold. Each record appears in `data/processed/manifest.csv` as accepted or rejected with a reason. Accepted matrices are stored in `dataset_matrices.npz`; summary counts are in `dataset_summary.json`.

```powershell
python -m backend.dataset --workers 4
```

Processing saves each 100-record batch atomically under `data/processed/checkpoints/`. Four workers validate structures in parallel by default; lower `--workers` if memory is limited. If the command is interrupted or the computer shuts down, run it again after startup: completed batches are reused and processing resumes at the first incomplete batch. The checkpoint key changes if an input file, threshold, seed, or batch size changes. Pass `--fresh` only when you intentionally want to recompute all batches.

Train and evaluate the CNN. This also runs stratified group cross-validation on the training portion only. The model and actual results are written to `models/cnn_20x20_three_conv.keras` and `data/processed/`.

```powershell
python backend/ml_pipeline.py --folds 10 --epochs 100
```

Completed cross-validation folds are recorded in a fingerprinted checkpoint file. Keras `BackupAndRestore` also saves the in-progress fold and final training state by epoch. An interrupted command can be rerun with the same options to resume; changing training inputs or options starts a separate checkpoint run.

Start the web interface after training:

```powershell
python backend/app.py
```

This starts the FastAPI application with Uvicorn. Open `http://127.0.0.1:5000`. Enter a PDB ID or upload a PDB/mmCIF file; do not provide both. The app finds contacting protein-chain pairs automatically. PDB-ID lookup requires an internet connection to the RCSB file service. A trained model must exist before predictions can run. To launch Uvicorn directly, use `python -m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 5000`.

## CNN and evaluation

The CNN input is a `20 × 20 × 1` interaction matrix. It has three Conv2D layers with 32 filters and `3 × 3` kernels, each followed by MaxPooling2D, then Flatten, Dense(512), Dense(64), and a one-neuron sigmoid output. Conv2D uses same padding; MaxPooling2D uses its Keras default padding. Binary cross-entropy and Adam are used. Training-only class weights address the class imbalance without discarding supplied examples. Early stopping and best-model selection use validation loss. The test split is not passed to cross-validation or model fitting. Training and evaluation outputs must be regenerated after this architecture change.

TensorFlow uses a GPU automatically when the installed TensorFlow build can see one. Structure parsing and contact detection are CPU-bound; moving only model training to a GPU machine requires transferring `dataset_matrices.npz` and the matching report files.

Cross-validation defaults to 10 folds, matching the supplied paper, and uses `StratifiedGroupKFold` to preserve class proportions while keeping all records for a PDB identifier in one fold. If fewer than 10 independent groups per class pass validation, the code lowers the fold count to the supported number and reports it. Each fold starts from fresh CNN weights. Fold metrics and their mean/standard deviation are saved in `cross_validation_results.json`. Cross-validation uses only the training partition; the separate test partition remains untouched until final evaluation. The final test metrics, confusion matrix, training history, and plots are saved separately. ROC-AUC is omitted when a test partition contains only one class.

Accuracy is the overall fraction of correct test predictions. Precision describes how often positive predictions are correct. Recall describes how many true positive cases were found. F1 balances precision and recall. ROC-AUC measures how well positive cases are ranked above negative cases across thresholds. In the confusion matrix, rows are true labels and columns are predictions, ordered negative then positive.

## Prediction limits

The confidence score is a CNN probability, not experimental confirmation. The app generates candidates from individual contacting protein-chain pairs, predicts both partner orientations, averages their antibody–antigen probabilities, and reports the most antibody–antigen-like pair. This automatic pair scan is an inference aid; complexes with multichain partners may need specialized chain grouping for a more faithful analysis. A structure rejected by validation is never passed to the model, and no matrix is shown for a rejected structure. Same-amino-acid contact counts are omitted to preserve the required zero diagonal.
