import torch
import numpy as np
import pandas as pd

def score_metabolites(model, expr_mat, meta_ids, well_ids,
                      device="cpu", batch_size=64):

    model.eval()
    N = expr_mat.shape[0]
    latent_list = []

    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = expr_mat[i : i + batch_size]
            X = torch.tensor(batch, dtype=torch.float32).to(device)

            _, _, _, latent = model(X)
            latent_list.append(latent.cpu().numpy())

    # (N,latent_dim)
    latents = np.concatenate(latent_list,axis=0)

    # ---- per well ----
    df_well = pd.DataFrame({
        "Well": well_ids,
        "Metabolite": meta_ids,
        **{f"Latent_{i}": latents[:, i] for i in range(latents.shape[1])}
    })
    # Add a norm-based score as the final latent score
    df_well["LatentScore"] = np.linalg.norm(latents, axis=1)

    # ---- per metabolite ----
    df_meta = (
        df_well
        .groupby("Metabolite")["LatentScore"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "N"})
        .sort_values("mean", ascending=False)
    )

    return df_well, df_meta
