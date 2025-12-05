import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def _select_target(scores3, logits, act_scalar, latent, target: str, score_index: int):
    """
    Helper to pick the tensor to attribute.
    """
    if target == "logit":
        return logits  # (B,)
    if target == "act":
        return act_scalar  # (B,)
    if target == "scores3":
        if score_index < 0 or score_index >= scores3.shape[1]:
            raise ValueError(f"score_index {score_index} out of range for scores3 shape {scores3.shape}")
        return scores3[:, score_index]
    if target == "latent":
        if score_index < 0 or score_index >= latent.shape[1]:
            raise ValueError(f"score_index {score_index} out of range for latent shape {latent.shape}")
        return latent[:, score_index]
    raise ValueError(f"Unsupported target '{target}'. Use one of: logit, act, scores3, latent.")


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
        tgt_tensor = _select_target(scores3, logits, act_scalar, latent, target, score_index)

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


def compute_gene_importance_ig(model,
                               expr_mat: np.ndarray,
                               device: str = "cpu",
                               batch_size: int = 32,
                               target: str = "logit",
                               score_index: int = 0,
                               baseline: np.ndarray = None,
                               steps: int = 50):
    """
    Compute per-gene global importance using Integrated Gradients.

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
        Which output head to attribute ("logit"/"act"/"scores3"/"latent").
    score_index : int
        Index for "scores3" (0/1/2) or "latent" dimension.
    baseline : np.ndarray
        Baseline input of shape (G,). If None, uses zeros. For DMSO baseline,
        pass the mean DMSO expression vector.
    steps : int
        Number of interpolation steps for integrated gradients.

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

    # prepare baseline
    if baseline is None:
        base_vec = torch.zeros(n_genes, device=device, dtype=torch.float32)
    else:
        base_vec = torch.tensor(baseline, dtype=torch.float32, device=device)
        if base_vec.numel() != n_genes:
            raise ValueError(f"baseline length {base_vec.numel()} does not match n_genes {n_genes}")

    alphas = torch.linspace(0.0, 1.0, steps + 1, device=device)

    for batch in loader:
        (x,) = batch
        x = x.to(device)
        batch_size_actual = x.shape[0]
        base = base_vec.unsqueeze(0).expand_as(x)
        delta = x - base

        grad_sum = torch.zeros_like(x)

        for alpha in alphas[1:]:  # skip alpha=0 baseline
            x_interp = base + alpha * delta
            x_interp.requires_grad_(True)

            scores3, logits, act_scalar, latent = model(x_interp)
            tgt_tensor = _select_target(scores3, logits, act_scalar, latent, target, score_index)
            target_scalar = tgt_tensor.sum()

            model.zero_grad()
            if x_interp.grad is not None:
                x_interp.grad.zero_()

            target_scalar.backward()
            grad_sum += x_interp.grad.detach()

        # average gradient along path
        avg_grad = grad_sum / steps
        ig = delta * avg_grad  # (B, G)

        saliency_np = ig.abs().detach().cpu().numpy()  # (B, G)
        importance_sum += saliency_np.sum(axis=0)
        total_samples += batch_size_actual

    importance = importance_sum / float(total_samples)
    return importance
