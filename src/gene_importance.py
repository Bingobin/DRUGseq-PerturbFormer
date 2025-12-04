import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def compute_gene_importance(model,
                            expr_mat: np.ndarray,
                            device: str = "cpu",
                            batch_size: int = 32,
                            target: str = "logit",
                            score_index: int = 0):
    """
    Compute per-gene global importance using Gradient × Input.

    By default, attribution is taken w.r.t. the classification logit, which is
    the most consistently trained head. You can switch targets via `target`:
      - "logit": resting vs altered-state logit
      - "act": legacy scalar head (only meaningful if `act_old` was used in training)
      - "scores3": one of the three regression scores (pick with `score_index`)
      - "latent": one latent dimension (pick with `score_index`)

    Parameters
    ----------
    model : torch.nn.Module
        Trained GeneTransformerMultiTask model.
    expr_mat : np.ndarray
        Shape (N, G), same preprocessing as training (e.g. VST).
    device : str
        "cpu", "cuda", or "mps".
    batch_size : int
        Batch size for computing gradients.
    target : str
        Which output head to attribute. Defaults to "logit".
    score_index : int
        Index for "scores3" (0/1/2) or "latent" dimension.

    Returns
    -------
    importance : np.ndarray
        Shape (G,), global importance score for each gene.
    """

    model.eval()
    model.to(device)

    X = torch.tensor(expr_mat, dtype=torch.float32)
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    n_samples, n_genes = expr_mat.shape
    importance_sum = np.zeros(n_genes, dtype=np.float64)
    total_samples = 0

    for batch in loader:
        (x,) = batch
        x = x.to(device)
        x.requires_grad_(True)

        # forward
        scores3, logits, act_scalar, latent = model(x)

        # choose target tensor
        if target == "logit":
            tgt_tensor = logits  # (B,)
        elif target == "act":
            tgt_tensor = act_scalar  # (B,)
        elif target == "scores3":
            if score_index < 0 or score_index >= scores3.shape[1]:
                raise ValueError(f"score_index {score_index} out of range for scores3 shape {scores3.shape}")
            tgt_tensor = scores3[:, score_index]
        elif target == "latent":
            if score_index < 0 or score_index >= latent.shape[1]:
                raise ValueError(f"score_index {score_index} out of range for latent shape {latent.shape}")
            tgt_tensor = latent[:, score_index]
        else:
            raise ValueError(f"Unsupported target '{target}'. Use one of: logit, act, scores3, latent.")

        # scalar target = sum over batch
        target_scalar = tgt_tensor.sum()
        model.zero_grad()
        if x.grad is not None:
            x.grad.zero_()

        # backward
        target_scalar.backward()

        # grad wrt input x: same shape as x (B, G)
        grad_x = x.grad.detach()                # (B, G)
        saliency = (grad_x * x).abs()          # Gradient × Input

        # move to CPU numpy
        saliency_np = saliency.detach().cpu().numpy()   # (B, G)

        # accumulate
        batch_size_actual = saliency_np.shape[0]
        importance_sum += saliency_np.sum(axis=0)  # sum over batch dimension
        total_samples += batch_size_actual

        # cleanup
        x.requires_grad_(False)
        x.grad = None

    # mean over all samples
    importance = importance_sum / float(total_samples)
    return importance


def save_gene_importance(gene_names, importance, out_csv="gene_importance_gradinput.csv"):
    """
    Save gene importance scores to CSV.
    """
    import pandas as pd

    df = pd.DataFrame({
        "Gene": gene_names,
        "GradInputImportance": importance
    })
    # sort by importance descending
    df = df.sort_values("GradInputImportance", ascending=False)
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Gene importance saved to {out_csv}")
