#!/usr/bin/env python3

import time
import argparse
import torch
import pandas as pd
import os
import numpy as np

from src.utils_seed import set_seed
from src.data_loader import load_data
from src.train import train_model
from src.scorer import score_metabolites
from src.plot_loss import plot_loss
from src.export_results import export_per_well

from src.model import GeneTransformerMultiTask
from src.gene_importance import compute_gene_importance, save_gene_importance
from src.latent import compute_latent_scores


def parse_args():
    parser = argparse.ArgumentParser(description="DrugSeq Transformer Full Pipeline")
    parser.add_argument("--seed", type=int, default=614,
                        help="Random seed (default: 614)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--use_zscore", action="store_true",
                        help="Apply z-score normalization to expression matrix (default: False)")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory containing input CSV files")
    parser.add_argument("--num_threads", type=int, default=10,
                        help="threads for torch on CPU")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training (default: 32)")
    parser.add_argument("--model_path", type=str, default="results/model.pt",
                        help="Path to model state_dict (for loading / saving)")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training and load existing model from --model_path")
    parser.add_argument("--gi_batch_size", type=int, default=32,
                        help="Batch size for Gradient×Input gene-importance (default: 32)")
    parser.add_argument("--out_dir", type=str, default="results",
                        help="Output directory for all results (default: results)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("DrugSeq Transformer Full Pipeline")
    print("=" * 80)

    print(f"[INFO] Setting random seed: {args.seed}")
    set_seed(args.seed)

    # device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        torch.set_num_threads(args.num_threads)

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Threads for torch on CPU: {torch.get_num_threads()}")

    # ensure output directory
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[INFO] Output directory: {args.out_dir}")

    # 1. Load data
    print("\n[1/6] Loading data...")
    t0 = time.time()
    expr, scores3, labels, meta, well, act_old, cluster = load_data(args.data_dir)
    print(f"[INFO] Data loaded: {expr.shape[0]} samples, {expr.shape[1]} genes")
    print(f"[INFO] activation_score_old available: {'Yes' if act_old is not None else 'No'}")
    print(f"[INFO] Time: {time.time() - t0:.2f} sec\n")

    expr_path = f"{args.data_dir}/expr_mat.csv"
    df_expr = pd.read_csv(expr_path)
    gene_names = df_expr.columns.to_list()

    # 2. Train or load model
    model = None
    history = None

    if not args.skip_train:
        print(f"[2/6] Training Transformer model for {args.epochs} epochs...")
        t0 = time.time()

        # Note: `train_model` returns 5 values here
        model, history, scaler, best_state, last_state = train_model(
            expr, scores3, labels, meta, well,
            act_old=act_old,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            use_zscore=args.use_zscore
        )

        print(f"[INFO] Training completed. Time: {time.time() - t0:.2f} sec\n")

        # Save scaler
        df_scaler = pd.DataFrame({
            "gene": gene_names,
            "mean": scaler["mean"],
            "std":  scaler["std"]
        })
        df_scaler.to_csv(f"{args.out_dir}/gene_scaler.csv", index=False)
        print(f"[INFO] Gene scaler saved to gene_scaler.csv")

        # 2.1 Plot loss curve
        if history is not None:
            df_hist = pd.DataFrame({
                "epoch": range(1, len(history["train_loss"]) + 1),
                "train_total": history["train_loss"],
                "val_total":   history["val_loss"],
                "train_mse":   history["train_mse"],
                "train_bce":   history["train_bce"],
                "train_reg":   history["train_reg"],
                "val_mse":     history["val_mse"],
                "val_bce":     history["val_bce"],
                "val_reg":     history["val_reg"],
            })
            df_hist.to_csv(f"{args.out_dir}/history_loss_detailed.csv", index=False)
            print("[INFO] Training loss history saved to history_loss_detailed.csv")

            print("[INFO] Plotting loss curve to loss_curve.png ...")
            plot_loss(history, out_file=f"{args.out_dir}/loss_curve.png")
            print("[INFO] Loss curve saved.\n")

        # 3. Save models
        print("[3/6] Saving models ...")
        # By default `model.pt` is the best model
        if best_state is not None:
            torch.save(best_state, f"{args.out_dir}/model_best.pt")
            torch.save(best_state, f"{args.out_dir}/model.pt")
            print("[INFO] Best model saved to model_best.pt and model.pt")
        else:
            # This should not happen in normal operation, but save current model just in case
            torch.save(model.state_dict(), f"{args.out_dir}/model.pt")
            print("[WARN] best_state is None, saved current model to model.pt")

        if last_state is not None:
            torch.save(last_state, f"{args.out_dir}/model_last.pt")
            print("[INFO] Last-epoch model saved to model_last.pt")

        print("[INFO] Model saving done.\n")

    else:
        # skip training: load model from disk (assumed to be the best model by default)
        print(f"[2/6] Skipping training. Loading model from {args.model_path} ...")
        n_genes = expr.shape[1]
        model = GeneTransformerMultiTask(n_genes=n_genes).to(device)
        state = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state)
        print("[INFO] Model loaded.\n")

    # 4. Scoring metabolites (per-well + per-metabolite)
    print("[4/6] Scoring metabolites (per-well + per-metabolite)...")
    t0 = time.time()
    from src.scorer import score_metabolites as score_metas  # ensure using the updated version
    df_well, df_meta = score_metas(
        model,
        expr_mat=expr,
        meta_ids=meta,
        well_ids=well,
        device=device,
        batch_size=64
    )

    df_well.to_csv(f"{args.out_dir}/well_activation_socres.csv", index=False)
    df_meta.to_csv(f"{args.out_dir}/metabolite_activation_scores.csv", index=False)

    print("[INFO] Top 5 metabolites by latent score:")
    print(df_meta.head())
    print("[INFO] Saved well_activation_socres.csv and metabolite_activation_scores.csv")

    print("[INFO] Exporting per-well predictions to well_predictions_results.csv ...")
    export_per_well(
        model, expr, scores3, labels, meta, well,
        device=device,
        outfile=f"{args.out_dir}/well_predictions_results.csv"
    )
    print(f"[INFO] Time: {time.time() - t0:.2f} sec\n")

    # 5. Gene importance (Gradient × Input)
    print("[5/6] Computing gene importance (Gradient × Input)...")
    t0 = time.time()

    # Re-read expr CSV to get column names (we already read it earlier, but this is fine)
    expr_path = f"{args.data_dir}/expr_mat.csv"
    df_expr = pd.read_csv(expr_path)
    gene_names = df_expr.columns.to_list()

    from src.gene_importance import compute_gene_importance, save_gene_importance

    importance = compute_gene_importance(
        model=model,
        expr_mat=expr,
        device=device,
        batch_size=args.gi_batch_size
    )

    save_gene_importance(
        gene_names=gene_names,
        importance=importance,
        out_csv=f"{args.out_dir}/gene_importance_gradinput.csv"
    )

    print(f"[INFO] Gene importance saved to gene_importance_gradinput.csv")
    print(f"[INFO] Time: {time.time() - t0:.2f} sec\n")

    # 6. Compute latent distance scores
    print("[6/6] Computing latent distance scores (relative to DMSO center)...")
    t0 = time.time()

    df_latent = compute_latent_scores(
        model=model,
        expr=expr,
        meta_ids=meta,
        well_ids=well,
        device=device,
        batch_size=64
    )

    df_latent.to_csv(f"{args.out_dir}/latent_scores.csv", index=False)

    print("[INFO] Saved latent_scores.csv")
    print(f"[INFO] Time: {time.time() - t0:.2f} sec\n")

    print("=" * 80)
    print("Pipeline Completed. Outputs in:", args.out_dir)
    print("  - model.pt (best model)")
    print("  - model_best.pt")
    print("  - model_last.pt")
    print("  - metabolite_activation_scores.csv")
    print("  - well_activation_scores.csv")
    print("  - well_predictions_results.csv")
    print("  - gene_importance_gradinput.csv")
    print("  - latent_scores.csv")
    if history is not None:
        print("  - loss_curve.png")
        print("  - history_loss_detailed.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
