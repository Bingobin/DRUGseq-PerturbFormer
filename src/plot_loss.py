import matplotlib.pyplot as plt
import numpy as np

def plot_loss(history, out_file="loss_curve.png"):
    """
    Plot loss curves from history dict.

    Expected keys in history:
        - "train_total"
        - "val_total"
        - (optional) "train_mse", "val_mse"
        - (optional) "train_bce", "val_bce"
        - (optional) "train_reg", "val_reg"

    The function automatically plots all keys that exist.
    """

    plt.figure(figsize=(8, 6))

    # Loop all keys dynamic
    for key, values in history.items():
        if not isinstance(values, list):
            continue
        if len(values) == 0:
            continue

        epochs = range(1, len(values) + 1)
        plt.plot(epochs, values, label=key)

    # If validation loss is available, mark the best epoch (minimum val loss)
    if "val_loss" in history and isinstance(history["val_loss"], list) and len(history["val_loss"]) > 0:
        val_losses = history["val_loss"]
        best_idx = int(np.argmin(val_losses))
        best_epoch = best_idx + 1
        best_val = val_losses[best_idx]

        # draw vertical line at best epoch
        plt.axvline(best_epoch, color="red", linestyle="--", alpha=0.6)

        # mark the val_loss point
        plt.scatter([best_epoch], [best_val], color="red", zorder=10)

        # annotate with text above the point
        annot_text = f"Best epoch: {best_epoch}\nval_loss={best_val:.4f}"
        plt.annotate(annot_text,
                     xy=(best_epoch, best_val),
                     xytext=(0, 10),
                     textcoords="offset points",
                     ha="center",
                     va="bottom",
                     fontsize=9,
                     color="red",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        # optionally mark training loss at same epoch if present
        if "train_loss" in history and isinstance(history["train_loss"], list) and len(history["train_loss"]) > best_idx:
            train_val = history["train_loss"][best_idx]
            plt.scatter([best_epoch], [train_val], color="blue", zorder=9)
            plt.annotate(f"train={train_val:.4f}",
                         xy=(best_epoch, train_val),
                         xytext=(0, -12),
                         textcoords="offset points",
                         ha="center",
                         va="top",
                         fontsize=8,
                         color="blue",
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend(loc="best", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"[INFO] Loss curve saved to {out_file}")
