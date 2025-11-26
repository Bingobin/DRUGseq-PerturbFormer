import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from .dataset import DrugSeqDataset
from .model import GeneTransformerMultiTask


def create_loaders(expr, scores, labels, meta, well,
                   batch_size=32, val_size=0.2, seed=614,
                   use_zscore=False):

    N = expr.shape[0]
    indices = list(range(N))

    # deterministic split
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_size,
        stratify=labels,
        random_state=seed
    )
    
    # Compute train-set mean/std for scaler output regardless of normalization choice
    train_expr = expr[train_idx]
    gene_mean = train_expr.mean(axis=0)
    gene_std = train_expr.std(axis=0) + 1e-8

    if use_zscore:
        # Apply z-score normalization to inputs
        expr_z = (expr - gene_mean) / gene_std
    else:
        # Keep original expression values for training input
        expr_z = expr

    train_dataset = DrugSeqDataset(expr_z, scores, labels, meta, well, train_idx)
    val_dataset   = DrugSeqDataset(expr_z, scores, labels, meta, well, val_idx)

    # optional: deterministic DataLoader shuffle
    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, generator=g)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size,
                              shuffle=False)

    return train_loader, val_loader, gene_mean, gene_std


def train_model(expr, scores, labels, meta, well,
                act_old=None,
                device="cpu",
                epochs=50,
                batch_size=32,
                seed=614,
                use_zscore=False):
    """
    训练多任务 Transformer。
    返回：
      - model: 已加载 best_state 的模型（用于后续分析）
      - history: 各种 loss 曲线
      - scaler: 基因均值/方差
      - best_state: 验证集 total loss 最低 epoch 的 state_dict
      - last_state: 最后一个 epoch 的 state_dict
    """

    print("[INFO] Initializing model...")
    model = GeneTransformerMultiTask(n_genes=expr.shape[1]).to(device)

    train_loader, val_loader, gene_mean, gene_std = create_loaders(
        expr, scores, labels, meta, well,
        batch_size=batch_size,
        val_size=0.2,
        seed=seed,
        use_zscore=use_zscore
    )

    print(f"[INFO] Training samples: {len(train_loader.dataset)}")
    print(f"[INFO] Validation samples: {len(val_loader.dataset)}")

    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    use_reg = act_old is not None
    if use_reg:
        act_old = torch.tensor(act_old, dtype=torch.float32)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # history dict to store losses
    history = {
        "train_loss": [],
        "val_loss":   [],
        "train_mse":  [],
        "train_bce":  [],
        "train_reg":  [],
        "val_mse":    [],
        "val_bce":    [],
        "val_reg":    [],
    }

    # ==== 新增：追踪 best / last ====
    best_val = float("inf")
    best_state = None
    best_epoch = -1
    last_state = None
    # ============================

    for ep in range(1, epochs + 1):
        print("-" * 60)
        print(f"Epoch {ep}/{epochs}")

        # ---- train ----
        model.train()
        train_total = train_mse = train_bce = train_reg = 0.0

        for batch_idx, (x, y3, y_cls, idx, meta_i, well_i) in enumerate(train_loader):
            x = x.to(device)
            y3 = y3.to(device)
            y_cls = y_cls.to(device)

            optimizer.zero_grad()
            y3_pred, logits, act_lat, latent = model(x)

            # 分别计算三个部分
            loss_mse = mse(y3_pred, y3)
            loss_bce = bce(logits, y_cls)

            if use_reg:
                loss_reg = mse(act_lat, act_old[idx].to(device))
                loss = loss_mse + loss_bce + 0.2 * loss_reg
            else:
                loss_reg = torch.tensor(0.0, device=device)
                loss = loss_mse + loss_bce

            loss.backward()
            optimizer.step()

            train_total += loss.item()
            train_mse   += loss_mse.item()
            train_bce   += loss_bce.item()
            train_reg   += loss_reg.item()

            if batch_idx % 4 == 0:
                print(f"  Training batch {batch_idx}/{len(train_loader)}, "
                      f"loss={loss.item():.4f}")

        n_train_batches = len(train_loader)
        avg_train_total = train_total / n_train_batches
        avg_train_mse   = train_mse   / n_train_batches
        avg_train_bce   = train_bce   / n_train_batches
        avg_train_reg   = train_reg   / n_train_batches

        print(f"  Training loss - total={avg_train_total:.4f}, "
              f"mse={avg_train_mse:.4f}, bce={avg_train_bce:.4f}, reg={avg_train_reg:.4f}")

        # ---- validation ----
        model.eval()
        val_total = val_mse = val_bce = val_reg = 0.0

        with torch.no_grad():
            for x, y3, y_cls, idx, meta_i, well_i in val_loader:
                x = x.to(device)
                y3 = y3.to(device)
                y_cls = y_cls.to(device)

                y3_pred, logits, act_lat, latent = model(x)

                loss_mse = mse(y3_pred, y3)
                loss_bce = bce(logits, y_cls)

                if use_reg:
                    loss_reg = mse(act_lat, act_old[idx].to(device))
                    loss = loss_mse + loss_bce + 0.2 * loss_reg
                else:
                    loss_reg = torch.tensor(0.0, device=device)
                    loss = loss_mse + loss_bce

                val_total += loss.item()
                val_mse   += loss_mse.item()
                val_bce   += loss_bce.item()
                val_reg   += loss_reg.item()

        n_val_batches = len(val_loader)
        avg_val_total = val_total / n_val_batches
        avg_val_mse   = val_mse   / n_val_batches
        avg_val_bce   = val_bce   / n_val_batches
        avg_val_reg   = val_reg   / n_val_batches

        print(f"  Validation loss - total={avg_val_total:.4f}, "
              f"mse={avg_val_mse:.4f}, bce={avg_val_bce:.4f}, reg={avg_val_reg:.4f}")

        # record into history
        history["train_loss"].append(avg_train_total)
        history["val_loss"].append(avg_val_total)

        history["train_mse"].append(avg_train_mse)
        history["train_bce"].append(avg_train_bce)
        history["train_reg"].append(avg_train_reg)

        history["val_mse"].append(avg_val_mse)
        history["val_bce"].append(avg_val_bce)
        history["val_reg"].append(avg_val_reg)

        # ==== 更新 last_state ====
        # 注意：state_dict() 返回的是参数拷贝，不是引用
        last_state = model.state_dict()

        # ==== 更新 best_state ====
        if avg_val_total < best_val:
            best_val = avg_val_total
            best_state = model.state_dict()
            best_epoch = ep
            print(f"  [BEST] Updated best model at epoch {ep} (val_total={best_val:.4f})")

    # ==== 训练结束后，加载 best_state 到 model ====
    if best_state is not None:
        print(f"[INFO] Loading best model from epoch {best_epoch} "
              f"(val_total={best_val:.4f})")
        model.load_state_dict(best_state)

    scaler = {
        "mean": gene_mean,
        "std": gene_std
    }

    # 返回 best_model（model 对象）、history、scaler，以及两个 state_dict
    return model, history, scaler, best_state, last_state
