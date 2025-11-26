import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def export_per_well(
        model, expr, scores3, labels, meta, well,
        device="cpu", outfile="per_well_results.csv",
        batch_size=64):

    model.eval()
    model.to(device)

    X = torch.tensor(expr, dtype=torch.float32)
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # storage
    y3_list = []
    act_list = []
    latent_list = []

    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)

            # 新模型：scores3, logits, act_scalar, latent_vector
            y3_pred, _, act_scalar, latent = model(x)

            y3_list.append(y3_pred.cpu().numpy())
            act_list.append(act_scalar.cpu().numpy())
            latent_list.append(latent.cpu().numpy())

    # concatenate outputs
    y3_all  = np.concatenate(y3_list, axis=0)
    act_all = np.concatenate(act_list, axis=0)
    latent_all = np.concatenate(latent_list, axis=0)    # (N, latent_dim)

    df = pd.DataFrame({
        "well_id": well,
        "metabolite": meta,
        "label_cls": labels,
        "Tscore_z": scores3[:, 0],
        "CytoTRACE_z": scores3[:, 1],
        "Pseudotime_z": scores3[:, 2],
        "pred_Tscore": y3_all[:, 0],
        "pred_CytoTRACE": y3_all[:, 1],
        "pred_Pseudotime": y3_all[:, 2],
        "pred_activation": act_all,
        "latent_vector": [v.tolist() for v in latent_all],
    })

    df.to_csv(outfile, index=False)
    print(f"[INFO] per-well results saved to {outfile}")
