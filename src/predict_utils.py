import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .model import GeneTransformerMultiTask


def load_arrays(expr_csv, meta_csv=None, well_csv=None):
    """
    Load expression, optional meta/well IDs.
    """
    df_expr = pd.read_csv(expr_csv)
    expr = df_expr.values.astype(np.float32)

    if meta_csv and os.path.exists(meta_csv):
        meta = pd.read_csv(meta_csv).values.squeeze().astype(str)
    else:
        meta = np.array([None] * expr.shape[0], dtype=object)

    if well_csv and os.path.exists(well_csv):
        well = pd.read_csv(well_csv).values.squeeze().astype(str)
    else:
        well = np.array([f"sample_{i}" for i in range(expr.shape[0])], dtype=object)

    return df_expr.columns.to_list(), expr, meta, well


def load_model(model_path, n_genes, device, latent_dim=None):
    state = torch.load(model_path, map_location=device)

    # infer latent_dim from checkpoint if not provided
    if latent_dim is None:
        if "head_latent.weight" in state:
            latent_dim = state["head_latent.weight"].shape[0]
        else:
            latent_dim = 32  # fallback to default

    model = GeneTransformerMultiTask(n_genes=n_genes, latent_dim=latent_dim).to(device)
    model.load_state_dict(state)
    return model


def run_inference(model, expr, device="cpu", batch_size=64):
    model.eval()
    model.to(device)
    X = torch.tensor(expr, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)

    preds3, logits_list, latents = [], [], []
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            y3_pred, logits, _, latent = model(x)
            preds3.append(y3_pred.cpu().numpy())
            logits_list.append(logits.cpu().numpy())
            latents.append(latent.cpu().numpy())

    y3 = np.concatenate(preds3, axis=0)
    logits = np.concatenate(logits_list, axis=0).reshape(-1)
    latents = np.concatenate(latents, axis=0)
    probs = 1.0 / (1.0 + np.exp(-logits))
    classes = (probs >= 0.5).astype(int)
    return y3, logits, probs, classes, latents


def compute_latent_distances(latents, meta=None, dmso_meta="DMSO"):
    """
    Compute Euclidean and cosine distance to the DM SO center (or global mean if DMSO missing).
    """
    if meta is not None and (meta == dmso_meta).any():
        mu_dmso = latents[meta == dmso_meta].mean(axis=0)
    else:
        mu_dmso = latents.mean(axis=0)

    diff = latents - mu_dmso
    euclid = np.sqrt((diff ** 2).sum(axis=1))

    mu_norm = np.linalg.norm(mu_dmso) + 1e-8
    lat_norm = np.linalg.norm(latents, axis=1) + 1e-8
    cos_sim = (latents @ mu_dmso) / (lat_norm * mu_norm)
    cos_dist = 1.0 - cos_sim
    return euclid, cos_dist


def build_output_dataframe(well, meta, y3, logits, probs, classes, latents, euclid, cos_dist, latent_index=None):
    out = {
        "Well": well,
        "Metabolite": meta,
        "Pred_Tscore": y3[:, 0],
        "Pred_CytoTRACE": y3[:, 1],
        "Pred_Pseudotime": y3[:, 2],
        "Pred_Logit": logits,
        "Pred_Prob": probs,
        "Pred_Class": classes,
        "Latent_Euclid": euclid,
        "Latent_Cosine": cos_dist,
    }

    for i in range(latents.shape[1]):
        out[f"Latent_{i}"] = latents[:, i]

    if latent_index is not None and 0 <= latent_index < latents.shape[1]:
        out["Latent_Selected"] = latents[:, latent_index]

    return pd.DataFrame(out)
