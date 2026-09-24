from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_PATH = "plot_generator/data/scatter_data.csv"
OUTPUT_PATH = "Fig_13.pdf"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
})


def save_scatter_grid(y_true, predictions):
    if y_true.size == 0 or not np.isfinite(y_true).all():
        raise ValueError("Empty or non-finite y_true values.")
    for label, y_pred in predictions.items():
        if y_pred.shape != y_true.shape or not np.isfinite(y_pred).all():
            raise ValueError(f"Invalid predictions for {label}.")
    values = np.concatenate([y_true, *predictions.values()])
    lower, upper = float(values.min()), float(values.max())
    padding = 0.04 * (upper - lower) if upper > lower else 0.1
    limits = (lower - padding, upper + padding)

    with plt.rc_context({"font.family": "serif", "font.size": 12}):
        fig, axes = plt.subplots(2, 3, figsize=(14, 10), sharex=True, sharey=True)
        for index, (ax, (label, y_pred)) in enumerate(zip(axes.flat, predictions.items())):
            ax.scatter(y_true, y_pred, s=12, alpha=0.6, edgecolors="none")
            ax.plot(limits, limits, color="black", linewidth=1)
            ax.set(xlim=limits, ylim=limits, aspect="equal")
            ax.set_xlabel(r"$c_D$", fontsize=14)
            ax.set_ylabel(r"$\hat{c}_D$", fontsize=14)
            ax.set_title(f"({chr(ord('a') + index)}) {label}", fontsize=16)
            ax.tick_params(labelbottom=True, labelleft=True)
            ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight")
        plt.close(fig)


def main():
    frame = pd.read_csv(DATA_PATH)
    y_true = frame["y_true"].to_numpy(dtype=float)
    predictions = {
        name: frame[name].to_numpy(dtype=float)
        for name in frame.columns
        if name != "y_true"
    }
    save_scatter_grid(y_true, predictions)


if __name__ == "__main__":
    main()
