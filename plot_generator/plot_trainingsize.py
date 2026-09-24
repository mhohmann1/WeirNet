import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
})

def plot_r2_by_fraction(metrics_df, output_path, metric):
    label_size = 14
    tick_size = 12
    legend_size = 11
    color_cycle = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#9467bd",  # purple
        "#2ca02c",  # green
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # olive
        "#17becf",  # cyan
        "#d62728",  # red
    ]
    if metric not in metrics_df.columns:
        raise ValueError(f"Metric '{metric}' not found in metrics table.")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_prop_cycle(color=color_cycle)
    for model_name, group in metrics_df.groupby("model"):
        group = group.sort_values(by="train_fraction")
        x_vals = group["train_fraction"].to_numpy() * 100.0
        y_vals = group[metric].to_numpy()
        ax.plot(x_vals, y_vals, marker="o", linewidth=1.5, label=model_name)

    ax.set_xlabel("Training data (%)", fontsize=label_size)
    ax.set_ylabel(r"$R^2$", fontsize=label_size)
    ax.grid(True, alpha=0.8)
    ax.tick_params(axis="both", labelsize=tick_size)
    ax.legend(fontsize=legend_size)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)

metrics = pd.read_csv("plot_generator/data/metrics.csv")
plot_r2_by_fraction(metrics, "Fig_16.pdf", "test_r2")
