#!/usr/bin/env python3

import argparse
import os
import torch

from src.predict_utils import (
    load_arrays,
    load_model,
    run_inference,
    compute_latent_distances,
    build_output_dataframe,
)


def parse_args():
    p = argparse.ArgumentParser(description="Predict on new DRUG-seq / bulk RNA-seq samples")
    p.add_argument("--expr_csv", required=True, help="CSV with expression matrix; columns must match training genes")
    p.add_argument("--meta_csv", default=None, help="Optional CSV with metabolite IDs (N,)")
    p.add_argument("--well_csv", default=None, help="Optional CSV with well/sample IDs (N,)")
    p.add_argument("--model_path", default="results/model.pt", help="Path to trained model state_dict")
    p.add_argument("--out_csv", default="results/predict_results.csv", help="Output CSV for predictions")
    p.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"], help="Device override")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
    p.add_argument("--dmso_meta", type=str, default="DMSO", help="Metabolite ID used to define baseline center")
    p.add_argument("--latent_index", type=int, default=None, help="Optional latent dimension to highlight as Latent_Selected")
    return p.parse_args()


def main():
    args = parse_args()

    # device selection
    if args.device:
        device = args.device
    else:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    _, expr, meta, well = load_arrays(args.expr_csv, args.meta_csv, args.well_csv)

    # load model
    model = load_model(args.model_path, n_genes=expr.shape[1], device=device)

    # forward pass
    y3, logits, probs, classes, latents = run_inference(model, expr, device, args.batch_size)

    # distances
    euclid, cos_dist = compute_latent_distances(latents, meta, args.dmso_meta)

    # build output
    df_out = build_output_dataframe(
        well=well,
        meta=meta,
        y3=y3,
        logits=logits,
        probs=probs,
        classes=classes,
        latents=latents,
        euclid=euclid,
        cos_dist=cos_dist,
        latent_index=args.latent_index
    )

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True) if os.path.dirname(args.out_csv) else None
    df_out.to_csv(args.out_csv, index=False)
    print(f"[INFO] Saved predictions to {args.out_csv}")


if __name__ == "__main__":
    main()
