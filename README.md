# Tcell_DRUGseq_transformer

English overview of the DRUG-seq multi-task Transformer pipeline for T cell activation profiling. The pipeline trains a transformer encoder on gene expression to jointly learn regression, classification, activation scoring, latent embeddings, and downstream analyses.

## What It Does
- Loads DRUG-seq expression matrix and metadata (`run_pipeline.py`, `src/data_loader.py`).
- Multi-task Transformer (`src/model.py`) that embeds gene IDs/values, encodes with a Transformer, and predicts:
  - Three continuous scores (`Tscore`, `CytoTRACE`, `Pseudotime`)
  - Binary classification logit
  - Scalar activation score
  - Latent vector
- Training loop with z-score option, best/last checkpoints, loss curves (`src/train.py`, `src/plot_loss.py`).
- Exports per-well predictions, metabolite latent scores, latent distances to DMSO, and Gradient×Input gene importance (`src/export_results.py`, `src/scorer.py`, `src/latent.py`, `src/gene_importance.py`).

## Key Formulas
- **Z-score normalization (optional)** for input genes:
  \[
  x_{i,g}^{(z)} = \frac{x_{i,g} - \mu_g}{\sigma_g + 10^{-8}}
  \]
  where \(\mu_g, \sigma_g\) are computed on the training split.

- **Multi-task loss** (per batch) combining regression, classification, and optional activation regularization:
  \[
  \mathcal{L} = \text{MSE}(y^{(3)}, \hat{y}^{(3)}) \;+\; \text{BCEWithLogits}(y^{(\text{cls})}, \hat{y}^{(\text{cls})})
  \;+\; \lambda_{\text{reg}} \,\text{MSE}(y^{(\text{act-old})}, \hat{y}^{(\text{act})})
  \]
  with \(\lambda_{\text{reg}} = 0.2\) when historical activation scores exist.

- **Latent norm score** per sample:
  \[
  s_{\text{latent}} = \lVert z \rVert_2
  \]
  Aggregated by metabolite (mean/std/count) for ranking.

- **Latent distance to DMSO center**:
  \[
  \mu_{\text{DMSO}} = \frac{1}{N_{\text{DMSO}}} \sum_{i \in \text{DMSO}} z_i,\quad
  d_{\text{euclid}}(i)=\lVert z_i - \mu_{\text{DMSO}}\rVert_2,\quad
  d_{\text{cos}}(i)=1-\frac{z_i \cdot \mu_{\text{DMSO}}}{\lVert z_i\rVert_2\,\lVert \mu_{\text{DMSO}}\rVert_2}
  \]

- **Gradient×Input gene importance** (global average over samples):
  \[
  \text{GI}_g = \frac{1}{N} \sum_{i=1}^{N} \left| x_{i,g} \,\frac{\partial \hat{a}_i}{\partial x_{i,g}} \right|
  \]
  where \(\hat{a}_i\) is the predicted activation scalar.

## Environment Setup
1. Python 3.9+ recommended. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies (for GPU, install the CUDA-matched PyTorch wheel first if needed):
   ```bash
   pip install -r requirements.txt
   ```

## Data Requirements
Expected CSVs under `data/`:
- `expr_mat.csv`: (N, G) expression matrix, columns are genes.
- `scores_3_z.csv`: (N, 3) continuous targets.
- `labels_cls.csv`: (N,) binary labels.
- `meta_id.csv`: (N,) metabolite/condition IDs.
- `well_id.csv`: (N,) well IDs.
- Optional: `activation_score_old.csv` (historical activation), `cluster_id.csv` (clusters, used to subset DMSO).

## Run Examples
Train from scratch and export all outputs:
```bash
python run_pipeline.py \
  --epochs 50 \
  --batch_size 32 \
  --data_dir data \
  --out_dir results
```
Common flags:
- `--skip_train`: skip training and load weights from `--model_path`.
- `--use_zscore`: enable z-score normalization of gene expression.
- `--seed`, `--num_threads`: control reproducibility and CPU threading.

Evaluate an existing model without training:
```bash
python run_pipeline.py --skip_train --model_path results/model.pt
```

## Outputs
Written to `--out_dir`:
- `model.pt` (best), `model_best.pt`, `model_last.pt`
- `well_activation_socres.csv`, `metabolite_activation_scores.csv`
- `well_predictions_results.csv`
- `gene_importance_gradinput.csv`
- `latent_scores.csv`
- If training: `loss_curve.png`, `history_loss_detailed.csv`, `gene_scaler.csv`

## Code Map
- `run_pipeline.py`: orchestrates the full flow.
- `src/model.py`: multi-task Transformer.
- `src/train.py`, `src/dataset.py`: loaders and training loop.
- `src/scorer.py`, `src/latent.py`: latent scoring and distances.
- `src/gene_importance.py`: Gradient×Input importance.
- `src/export_results.py`: per-well prediction export.
- `src/plot_loss.py`: loss visualization.
- `src/data_loader.py`, `src/utils_seed.py`: IO and seeding.
