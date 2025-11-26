import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def compute_gene_importance(model,
                            expr_mat: np.ndarray,
                            device: str = "cpu",
                            batch_size: int = 32):
    """
    Compute per-gene global importance using Gradient × Input w.r.t act_latent.

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
        _, _, act_scalar, _ = model(x)  # act_latent: (B,)

        # scalar target = sum over batch
        target = act_scalar.sum()
        model.zero_grad()
        if x.grad is not None:
            x.grad.zero_()

        # backward
        target.backward()

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
