# src/latent.py

import numpy as np
import torch
import pandas as pd


def compute_latent_vectors(model, expr_mat, device="cpu", batch_size=32):
    """
    Compute latent vectors for all samples.

    Returns
    -------
    latents : np.ndarray
        Shape (N, D), where D = latent_dim of the model.
    """
    model.eval()
    model.to(device)

    X = torch.tensor(expr_mat, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    latents = []
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            # 新模型: scores3, logits, act_scalar, latent_vector
            _, _, _, latent = model(x)          # latent: (B, D)
            latents.append(latent.cpu().numpy())

    latents = np.concatenate(latents, axis=0)   # (N, D)
    return latents


def compute_latent_scores(model, expr, meta_ids, well_ids,
                          cluster_ids=None, dmso_cluster="C6",
                          device="cpu", batch_size=32):
    """
    Compute distance in latent space relative to DMSO center.

    Returns
    -------
    df_latent : pd.DataFrame
        Columns: [Well, Meta, Latent_Euclid, Latent_Cosine]
    """
    # 1. latent vectors (N, D)
    latents = compute_latent_vectors(
        model=model,
        expr_mat=expr,
        device=device,
        batch_size=batch_size
    )

    # 2. DMSO center
    meta_ids = np.array(meta_ids)
    well_ids = np.array(well_ids)

    if cluster_ids is not None:
        cluster_ids = np.array(cluster_ids)
        is_dmso = (meta_ids == "DMSO") & (cluster_ids == dmso_cluster)
    else:
        is_dmso = (meta_ids == "DMSO")

    if not is_dmso.any():
        raise ValueError("No DMSO samples found in meta_ids, cannot compute DMSO center.")

    dmso_lat = latents[is_dmso]           # (N_dmso, D)
    mu_dmso = dmso_lat.mean(axis=0)       # (D,)

    # 3. Euclidean distance: ||z - mu_dmso||
    diff = latents - mu_dmso             # (N, D)
    euclid = np.sqrt((diff ** 2).sum(axis=1))   # (N,)

    # 4. Cosine distance: 1 - cos(z, mu_dmso)
    mu_norm = np.linalg.norm(mu_dmso) + 1e-8
    lat_norm = np.linalg.norm(latents, axis=1) + 1e-8
    cos_sim = (latents @ mu_dmso) / (lat_norm * mu_norm)   # (N,)
    cos_dist = 1.0 - cos_sim                               # (N,)

    df = pd.DataFrame({
        "Well": well_ids,
        "Meta": meta_ids,
        "Latent_Euclid": euclid,
        "Latent_Cosine": cos_dist
    })

    return df
