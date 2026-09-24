import math
import os
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
})


DATA_PATH = "plot_generator/data/combined_data.npz"
MODEL_DIR = Path("plot_generator/saved_model")
OUTPUT_DIR = "./"
SEED = 42
SHAP_SAMPLES = 500
MODELS = ["XGBoost", "RandomForest", "LightGBM", "GradientBoosting"]

def load_dataframe(path):
    if path.endswith(".npz"):
        loaded = np.load(path, allow_pickle=True)
        data = loaded["data"]
        columns = loaded["columns"]
        return pd.DataFrame(data, columns=columns)
    return pd.read_csv(path)

def compute_shap_values(model, X, seed=42, max_samples=500):
    X_sample = X.sample(n=min(max_samples, len(X)), random_state=seed)
    background = shap.sample(X_sample, min(500, len(X_sample)), random_state=seed)

    def model_fn(data):
        return model.predict(pd.DataFrame(data, columns=X.columns))

    explainer = shap.KernelExplainer(model_fn, background)
    shap_values = np.asarray(explainer.shap_values(X_sample))
    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = shap_values[..., 0]
    if shap_values.shape != X_sample.shape:
        raise ValueError(f"Unexpected SHAP shape: {shap_values.shape}; expected {X_sample.shape}.")
    return shap_values, X_sample



def _format_feature_label(name):
    if name == "Q":
        return r"$Q$"
    if name == "B_i":
        return r"$B_i$"
    if name == "B_o":
        return r"$B_o$"
    if name == "B":
        return r"$B$"
    if name == "Alpha_deg":
        return r"$\alpha$"
    if name == "Ts_stern":
        return r"$T_{s,2}$"
    if name == "Ts_dach":
        return r"$T_{s,3}$"
    if name == "W_o_u":
        return r"$W_{o,u}$"
    if name == "W_o_d":
        return r"$W_{o,d}$"
    return name


def save_shap_summary_grid(plots, output_dir):
    if not plots:
        return
    plot_path = os.path.join(output_dir, "Fig_14.pdf")

    importance = np.mean(
        [np.mean(np.abs(item["shap_values"]), axis=0) for item in plots], axis=0
    )
    feature_order = np.argsort(-importance, kind="stable")[:20]
    limit = max(
        float(np.max(np.abs(np.asarray(item["shap_values"])[:, feature_order])))
        for item in plots
    )
    limit = 1.05 * limit if limit > 0 else 1.0
    ncols = min(2, len(plots))
    nrows = math.ceil(len(plots) / ncols)
    cmap = shap.plots.colors.red_blue

    with plt.rc_context(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    ):
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(7 * ncols, (0.4 * len(feature_order) + 2) * nrows),
            squeeze=False,
        )
        axes = axes.ravel()
        for index, (ax, item) in enumerate(zip(axes, plots)):
            features = item["X"].iloc[:, feature_order].copy()
            features.columns = [_format_feature_label(c) for c in features.columns]
            plt.sca(ax)
            shap.summary_plot(
                np.asarray(item["shap_values"])[:, feature_order], features,
                show=False, sort=False, color_bar=False, plot_size=None,
                max_display=len(feature_order), cmap=cmap,
            )
            ax.set_title(item["label"])
            ax.set_xlim(-limit, limit)

            label = ""
            number = index + 1
            while number:
                number, remainder = divmod(number - 1, 26)
                label = chr(ord("a") + remainder) + label
            ax.annotate(
                f"({label})", xy=(0, 0.5),
                xycoords=("axes fraction", ax.title),
                ha="left", va="center", fontsize=14,
                annotation_clip=False,
            )

        for ax in axes[len(plots):]:
            ax.axis("off")
        fig.tight_layout(rect=(0, 0, 0.91, 1), h_pad=3, w_pad=3)
        colorbar_ax = fig.add_axes([0.93, 0.2, 0.015, 0.6])
        colorbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=cmap),
            cax=colorbar_ax, ticks=[0, 1],
        )
        colorbar.set_ticklabels(["Low", "High"])
        colorbar.set_label("Feature value")
        colorbar.ax.tick_params(length=0)
        colorbar.outline.set_visible(False)
        for artist in fig.findobj():
            if artist.get_rasterized():
                artist.set_rasterized(False)
        fig.savefig(plot_path, format="pdf", bbox_inches="tight")
        plt.close(fig)


def main():
    if SHAP_SAMPLES < 1:
        raise ValueError("SHAP_SAMPLES must be at least 1.")
    model_paths = [MODEL_DIR / f"{name}_train100.joblib" for name in MODELS]
    missing = [str(path) for path in model_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Trained models not found: " + ", ".join(missing))

    df = load_dataframe(DATA_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    feature_names = None
    X = None
    plots = []
    for name, path in zip(MODELS, model_paths):
        print(f"Loading model: {path}")
        model = joblib.load(path)
        names = getattr(model, "feature_names_in_", None)
        if names is None:
            raise ValueError(f"Model {path} has no saved feature names.")
        names = list(names)
        if feature_names is None:
            feature_names = names
            missing_features = [column for column in names if column not in df.columns]
            if missing_features:
                raise ValueError(f"Feature columns not found in data: {missing_features}")
            X = df[names].apply(pd.to_numeric, errors="coerce")
            X = X.replace([np.inf, -np.inf], np.nan).dropna()
            if X.empty:
                raise ValueError("No valid feature rows available for SHAP explanations.")
        elif names != feature_names:
            raise ValueError(f"Model {path} uses different features or feature ordering.")

        print(f"Computing SHAP values: {name} (train 100%)")
        shap_values, shap_X = compute_shap_values(
            model, X, seed=SEED, max_samples=SHAP_SAMPLES,
        )
        plots.append({"label": name, "shap_values": shap_values, "X": shap_X})
    save_shap_summary_grid(plots, OUTPUT_DIR)


if __name__ == "__main__":
    main()
