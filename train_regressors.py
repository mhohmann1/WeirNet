import argparse
import json
import math
import os

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import shap

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
})

DEFAULT_DROP_COLS = ['Modell', 'Pressure', 'h_O', 'h_t', 'H_t', 'B_b' ,'N_B_i', 'N_B_o', 'W_i_u', 'W_i_d', 'Cycles', 'Alpha_rad', 'W' , 'Ht_P', 'C_d', 'T_s', 'L', 'P', 'W_u'] 
# DEFAULT_DROP_COLS = ["Modell", "Pressure", "Alpha_rad", "h_0", "h_t", "H_t", "Ht_P"] 
DEFAULT_MODELS = ["XGBoost", "RandomForest", "LightGBM","GradientBoosting"]
def load_dataframe(path):
    if path.endswith(".npz"):
        loaded = np.load(path, allow_pickle=True)
        data = loaded["data"]
        columns = loaded["columns"]
        return pd.DataFrame(data, columns=columns)
    return pd.read_csv(path)


def parse_list(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_feature_target(df, target, feature_cols, drop_cols):
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in data.")
    if feature_cols:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns not found in data: {missing}")
        X = df[feature_cols].copy()
    else:
        drop_set = set(drop_cols or [])
        drop_set.add(target)
        existing_drop = [c for c in drop_set if c in df.columns]
        X = df.drop(columns=existing_drop)
    y = df[target].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = X.notna().all(axis=1) & y.notna()
    return X.loc[mask], y.loc[mask]


def build_model_builders(seed):
    builders = {
        "LinearRegression": lambda: Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]),
        "ridge": lambda: Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "lasso": lambda: Pipeline([("scale", StandardScaler()), ("model", Lasso(alpha=1e-4, max_iter=5000))]),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
            max_depth=None,
        ),
        "GradientBoosting": lambda: GradientBoostingRegressor(random_state=seed),
    }
    if XGBRegressor is not None:
        builders["XGBoost"] = lambda: XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
        )
    if LGBMRegressor is not None:
        builders["LightGBM"] = lambda: LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            random_state=seed,
            n_jobs=-1,
        )
    return builders


def evaluate(model, X, y):
    preds = model.predict(X)
    return evaluate_preds(y, preds)


def evaluate_preds(y, preds):
    y_array = np.asarray(y)
    denom = np.where(y_array == 0, 1e-8, y_array)
    rel_error = np.mean(np.abs(preds - y_array) / denom)
    max_abs_error = float(np.max(np.abs(preds - y_array))) if len(y_array) else np.nan
    return {
        "r2": r2_score(y, preds),
        # "mse": mean_squared_error(y, preds, squared=True),
        "mse": mean_squared_error(y, preds),
        "mae": mean_absolute_error(y, preds),
        "max_abs_error": max_abs_error,
        "relative_error": rel_error,
    }


def _unwrap_estimator(model):
    if isinstance(model, Pipeline):
        scaler = model.named_steps.get("scale")
        estimator = model.named_steps.get("model", model.steps[-1][1])
        return estimator, scaler
    return model, None


def compute_shap_values(model, X, feature_names, seed=42, max_samples=500):
    if len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=seed)
    else:
        X_sample = X

    X_used = X_sample.copy()
    background_size = min(500, len(X_used))
    background = shap.sample(X_used, background_size, random_state=seed)

    def model_fn(data):
        if isinstance(data, pd.DataFrame):
            return model.predict(data)
        data_df = pd.DataFrame(data, columns=feature_names)
        return model.predict(data_df)

    explainer = shap.KernelExplainer(model_fn, background)
    shap_values = explainer.shap_values(X_used)
    method = "kernel"
    X_for_shap = X_used

    shap_arr = np.asarray(shap_values)
    if shap_arr.ndim == 3:
        shap_arr = np.mean(shap_arr, axis=0)
    return shap_arr, X_for_shap, method


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


def save_shap_summary_grid(plots, output_dir, train_fraction):
    if not plots:
        return
    frac_tag = int(round(train_fraction * 100))
    plot_path = os.path.join(output_dir, f"shap_summary_train{frac_tag:03d}_grid.pdf")

    # Rank features by their mean importance across models for a common order.
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

        # Keep SHAP points and the shared colorbar as vectors for all sample sizes.
        for artist in fig.findobj():
            if artist.get_rasterized():
                artist.set_rasterized(False)
        fig.savefig(plot_path, format="pdf", bbox_inches="tight")
        plt.close(fig)

    print(f"Saved SHAP summary grid: {plot_path}")


def save_scatter_grid(plots, out_path, title=None):
    if not plots:
        return
    label_size = 14
    tick_size = 12
    title_size = 16
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    for ax, item in zip(axes, plots):
        y_true = np.asarray(item["y_true"])
        y_pred = np.asarray(item["y_pred"])
        ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="none")
        min_val = float(np.min([y_true.min(), y_pred.min()]))
        max_val = float(np.max([y_true.max(), y_pred.max()]))
        if min_val == max_val:
            min_val -= 1.0
            max_val += 1.0
        ax.plot([min_val, max_val], [min_val, max_val], color="black", linewidth=1)
        ax.set_xlabel(r"$c_D$", fontsize=label_size)
        ax.set_ylabel(r"$\hat{c}_D$", fontsize=label_size)
        ax.set_title(item.get("label", ""), fontsize=title_size)
        ax.tick_params(axis="both", labelsize=tick_size)
        ax.grid(True, linestyle="--", alpha=0.3)
    for ax in axes[len(plots) :]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=title_size)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def load_indices(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Split file not found: {path}")
    values = np.loadtxt(path, dtype=int)
    return np.atleast_1d(values)


def select_by_indices(X, y, indices):
    indices = np.atleast_1d(indices).astype(int)
    x_index = X.index.to_numpy()
    mask = np.isin(indices, x_index)
    ordered = indices[mask]
    if len(ordered) == 0:
        raise ValueError("No samples left after applying split indices.")
    return X.loc[ordered], y.loc[ordered]


def split_id_indices(id_indices, seed, train_ratio=0.8, val_ratio=0.1):
    id_indices = np.atleast_1d(id_indices).astype(int)
    if len(id_indices) == 1:
        return id_indices, id_indices

    rng = np.random.default_rng(seed)
    rng.shuffle(id_indices)
    train_end = int(train_ratio * len(id_indices))
    val_end = train_end + int(val_ratio * len(id_indices))
    train_indices = id_indices[:train_end]
    val_indices = id_indices[train_end:val_end]

    if len(train_indices) == 0:
        train_indices = id_indices[:1]
        val_indices = id_indices[1:2]
    if len(val_indices) == 0:
        val_indices = id_indices[-1:]
        train_indices = id_indices[:-1]

    return train_indices, val_indices


def split_by_q_range(X, q_min, q_max, seed, train_ratio=0.8, val_ratio=0.1):
    if "Q" not in X.columns:
        raise ValueError("Q column not found; cannot perform Q OOD split.")
    q_vals = X["Q"].to_numpy()
    q_min_val = -np.inf if q_min is None else q_min
    q_max_val = np.inf if q_max is None else q_max
    ood_mask = (q_vals >= q_min_val) & (q_vals <= q_max_val)

    ood_indices = X.index[ood_mask].to_numpy()
    id_indices = X.index[~ood_mask].to_numpy()

    if len(ood_indices) == 0:
        raise ValueError(f"exclude_q range produced 0 samples (min={q_min}, max={q_max}).")
    if len(id_indices) == 0:
        raise ValueError("exclude_q range covers entire dataset; no in-distribution samples left.")

    train_indices, val_indices = split_id_indices(
        id_indices, seed, train_ratio=train_ratio, val_ratio=val_ratio
    )
    return train_indices, val_indices, ood_indices


def split_by_alpha_deg_range(X, alpha_min, alpha_max, seed, train_ratio=0.8, val_ratio=0.1):
    if "Alpha_deg" not in X.columns:
        raise ValueError("Alpha_deg column not found; cannot perform alpha OOD split.")
    alpha_vals = X["Alpha_deg"].to_numpy()
    alpha_min_val = -np.inf if alpha_min is None else alpha_min
    alpha_max_val = np.inf if alpha_max is None else alpha_max
    ood_mask = (alpha_vals >= alpha_min_val) & (alpha_vals <= alpha_max_val)

    ood_indices = X.index[ood_mask].to_numpy()
    id_indices = X.index[~ood_mask].to_numpy()

    if len(ood_indices) == 0:
        raise ValueError(
            f"exclude_alpha range produced 0 samples (min={alpha_min}, max={alpha_max})."
        )
    if len(id_indices) == 0:
        raise ValueError("exclude_alpha range covers entire dataset; no in-distribution samples left.")

    train_indices, val_indices = split_id_indices(
        id_indices, seed, train_ratio=train_ratio, val_ratio=val_ratio
    )
    return train_indices, val_indices, ood_indices


def parse_fractions(value):
    raw = parse_list(value)
    if not raw:
        return []
    fractions = []
    for item in raw:
        try:
            frac = float(item)
        except ValueError as exc:
            raise ValueError(f"Invalid train fraction '{item}'.") from exc
        if not 0.0 < frac <= 1.0:
            raise ValueError(f"Train fraction must be in (0, 1], got {frac}.")
        fractions.append(frac)
    return sorted(set(fractions))


def compute_bin_edges(values, n_bins):
    values = np.asarray(values, dtype=float)
    if n_bins < 1:
        raise ValueError("r2_bins must be >= 1.")
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    if min_val == max_val:
        return np.array([min_val, max_val], dtype=float)
    return np.linspace(min_val, max_val, n_bins + 1)


def run_r2_sweep(
    X,
    y,
    feature,
    model_builders,
    model_names,
    train_fractions,
    seed,
    output_dir,
    n_bins,
    shuffle_fractions=False,
):
    if feature not in X.columns:
        raise ValueError(f"{feature} column not found; cannot run R^2 sweep.")
    edges = compute_bin_edges(X[feature].to_numpy(), n_bins)
    results = []
    for bin_idx in range(len(edges) - 1):
        range_min = float(edges[bin_idx])
        range_max = float(edges[bin_idx + 1])

        print(range_min, range_max)

        include_max = bin_idx == len(edges) - 2
        if include_max:
            ood_mask = (X[feature] >= range_min) & (X[feature] <= range_max)
        else:
            ood_mask = (X[feature] >= range_min) & (X[feature] < range_max)

        ood_indices = X.index[ood_mask].to_numpy()
        id_indices = X.index[~ood_mask].to_numpy()
        if len(ood_indices) == 0 or len(id_indices) == 0:
            continue

        train_idx, _ = split_id_indices(id_indices, seed)
        X_train = X.loc[train_idx]
        y_train = y.loc[train_idx]
        X_test = X.loc[ood_indices]
        y_test = y.loc[ood_indices]

        train_order = np.arange(len(X_train))
        if shuffle_fractions:
            rng = np.random.default_rng(seed)
            rng.shuffle(train_order)

        for name in model_names:
            builder = model_builders[name]
            for frac in train_fractions:
                X_train_sub, y_train_sub = get_fraction_subset(X_train, y_train, train_order, frac)
                model = builder()
                model.fit(X_train_sub, y_train_sub)
                if len(y_test) < 2:
                    test_r2 = np.nan
                else:
                    test_r2 = r2_score(y_test, model.predict(X_test))
                results.append(
                    {
                        "feature": feature,
                        "bin_index": bin_idx,
                        "range_min": range_min,
                        "range_max": range_max,
                        "range_mid": 0.5 * (range_min + range_max),
                        "n_test": len(y_test),
                        "model": name,
                        "train_fraction": frac,
                        "n_train": len(X_train_sub),
                        "test_r2": test_r2,
                    }
                )

    if not results:
        return pd.DataFrame()
    results_df = pd.DataFrame(results)
    for frac in train_fractions:
        subset = results_df[results_df["train_fraction"] == frac]
        if subset.empty:
            continue
        label_size = 14
        tick_size = 12
        title_size = 16
        fig, ax = plt.subplots(figsize=(10, 6))
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
        ax.set_prop_cycle(color=color_cycle)
        for name in model_names:
            model_df = subset[subset["model"] == name].sort_values("range_mid")
            if model_df.empty:
                continue
            ax.plot(model_df["range_mid"], model_df["test_r2"], marker="o", label=name)
        if feature == "Alpha_deg":
            label_name = r"$\alpha$ [°]"
        elif feature == "Q":
            label_name = r"Q [$\mathrm{m}^3/\mathrm{s}$]"
        ax.set_xlabel(label_name, fontsize=label_size)
        ax.set_ylabel(r"$R^2$", fontsize=label_size)
        ax.tick_params(axis="both", labelsize=tick_size)
        ax.grid(True, linestyle="--", alpha=0.8)
        ax.legend(fontsize=tick_size)
        
        # ax.set_title(f"R^2 vs {feature}", fontsize=title_size)
        fig.tight_layout()
        frac_tag = int(round(frac * 100))
        plot_path = os.path.join(
            output_dir,
            f"r2_vs_{feature.lower()}_train{frac_tag:03d}.png",
        )
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)

    return results_df


def get_fraction_subset(X, y, ordered_indices, fraction):
    if fraction >= 1.0:
        return X, y
    n_samples = max(1, int(round(len(ordered_indices) * fraction)))
    subset_idx = ordered_indices[:n_samples]
    return X.iloc[subset_idx], y.iloc[subset_idx]


def report_split_health(train_idx, val_idx, test_idx, total_count):
    train_idx = np.atleast_1d(train_idx).astype(int)
    val_idx = np.atleast_1d(val_idx).astype(int)
    test_idx = np.atleast_1d(test_idx).astype(int)

    def _dup_count(arr):
        return len(arr) - len(np.unique(arr))

    dup_train = _dup_count(train_idx)
    dup_val = _dup_count(val_idx)
    dup_test = _dup_count(test_idx)

    overlap_tv = len(np.intersect1d(train_idx, val_idx))
    overlap_tt = len(np.intersect1d(train_idx, test_idx))
    overlap_vt = len(np.intersect1d(val_idx, test_idx))

    min_idx = min(train_idx.min(), val_idx.min(), test_idx.min())
    max_idx = max(train_idx.max(), val_idx.max(), test_idx.max())

    print("Split check:")
    print(f"  train/val overlap: {overlap_tv}")
    print(f"  train/test overlap: {overlap_tt}")
    print(f"  val/test overlap: {overlap_vt}")
    print(f"  duplicates (train/val/test): {dup_train}/{dup_val}/{dup_test}")
    print(f"  index range: [{min_idx}, {max_idx}] vs dataset size {total_count}")

    if min_idx < 0 or max_idx >= total_count:
        print("  warning: split indices are out of range for current dataset.")
    if min_idx >= 1 and max_idx == total_count:
        print("  warning: indices look 1-based; expected 0-based.")


def main():
    parser = argparse.ArgumentParser(description="Train classic regression models to predict C_d.")
    parser.add_argument(
        "--data-path",
        default="./Data/PKW_Efficiency_Dataset/combined_data.npz",
        help="Path to CSV or combined_data.npz",
    )
    parser.add_argument("--output-dir", default="./regression_runs", help="Directory to write metrics and plots.")
    parser.add_argument("--target", default="C_d", help="Target column to predict.")
    parser.add_argument(
        "--drop-cols",
        default=",".join(DEFAULT_DROP_COLS),
        help="Comma-separated columns to drop when building features.",
    )
    parser.add_argument(
        "--feature-cols",
        default="",
        help="Comma-separated feature columns; overrides --drop-cols if set.",
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help=f"Comma-separated models to run. Options: {', '.join(DEFAULT_MODELS)}",
    )
    parser.add_argument("--split-random", action="store_true", help="Use random split instead of txt indices.")
    parser.add_argument("--train-indices", default="train_val_test/train_indices.txt", help="Train indices file.")
    parser.add_argument("--val-indices", default="train_val_test/val_indices.txt", help="Validation indices file.")
    parser.add_argument("--test-indices", default="train_val_test/test_indices.txt", help="Test indices file.")
    parser.add_argument("--exclude-q-min", type=float, default=None, help="Min Q to hold out for OOD test.")
    parser.add_argument("--exclude-q-max", type=float, default=None, help="Max Q to hold out for OOD test.")
    parser.add_argument(
        "--exclude-alpha-min",
        type=float,
        default=None,
        help="Min Alpha_deg to hold out for OOD test.",
    )
    parser.add_argument(
        "--exclude-alpha-max",
        type=float,
        default=None,
        help="Max Alpha_deg to hold out for OOD test.",
    )
    parser.add_argument(
        "--r2-vs-alpha",
        action="store_true",
        help="Sweep Alpha_deg bins and plot test R^2 on the held-out bin.",
    )
    parser.add_argument(
        "--r2-vs-q",
        action="store_true",
        help="Sweep Q bins and plot test R^2 on the held-out bin.",
    )
    parser.add_argument(
        "--r2-bins",
        type=int,
        default=19,
        help="Number of bins to use for R^2 sweep plots.",
    )
    parser.add_argument("--train-fractions",
        default="0.1,0.2,0.4,0.6,0.8,1.0",
        # default="1.0",
        help="Comma-separated fractions of the training set to use.",
    )
    parser.add_argument(
        "--shuffle-fractions",
        action="store_true",
        help="Shuffle training indices before sampling fractions.",
    )
    parser.add_argument("--test-size", type=float, default=0.1, help="Fraction for test split.")
    parser.add_argument("--val-size", type=float, default=0.1, help="Fraction for validation split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--shap-samples",
        type=int,
        default=500,
        help="Max samples to use when computing SHAP values per model.",
    )
    parser.add_argument(
        "--shap",
        action="store_true",
        help="Compute SHAP values and save a labeled summary PDF grid per training fraction.",
    )
    args = parser.parse_args()

    df = load_dataframe(args.data_path)
    feature_cols = parse_list(args.feature_cols)
    drop_cols = parse_list(args.drop_cols)
    model_names = parse_list(args.models)

    X, y = build_feature_target(df, args.target, feature_cols, drop_cols)
    train_fractions = parse_fractions(args.train_fractions)

    if args.r2_vs_alpha and args.r2_vs_q:
        raise ValueError("Set only one of --r2-vs-alpha or --r2-vs-q at a time.")
    sweep_feature = None
    if args.r2_vs_alpha:
        sweep_feature = "Alpha_deg"
    elif args.r2_vs_q:
        sweep_feature = "Q"

    if "Q" not in X.columns:
        raise ValueError("Input features must include 'Q' for flow rate.")

    use_q_exclusion = (args.exclude_q_min is not None) or (args.exclude_q_max is not None)
    use_alpha_exclusion = (args.exclude_alpha_min is not None) or (args.exclude_alpha_max is not None)
    if use_q_exclusion and use_alpha_exclusion:
        raise ValueError("Set only one of --exclude-q-* or --exclude-alpha-* at a time.")

    if use_alpha_exclusion:
        train_idx, val_idx, test_idx = split_by_alpha_deg_range(
            X, args.exclude_alpha_min, args.exclude_alpha_max, args.seed
        )
        X_train, y_train = select_by_indices(X, y, train_idx)
        X_val, y_val = select_by_indices(X, y, val_idx)
        X_test, y_test = select_by_indices(X, y, test_idx)
    elif use_q_exclusion:
        train_idx, val_idx, test_idx = split_by_q_range(
            X, args.exclude_q_min, args.exclude_q_max, args.seed
        )
        X_train, y_train = select_by_indices(X, y, train_idx)
        X_val, y_val = select_by_indices(X, y, val_idx)
        X_test, y_test = select_by_indices(X, y, test_idx)
    elif args.split_random:
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X, y, test_size=args.test_size + args.val_size, random_state=args.seed, shuffle=True
        )
        val_fraction = args.val_size / (args.test_size + args.val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp, test_size=1.0 - val_fraction, random_state=args.seed, shuffle=True
        )
    else:
        train_idx = load_indices(args.train_indices)
        val_idx = load_indices(args.val_indices)
        test_idx = load_indices(args.test_indices)
        report_split_health(train_idx, val_idx, test_idx, len(X))
        X_train, y_train = select_by_indices(X, y, train_idx)
        X_val, y_val = select_by_indices(X, y, val_idx)
        X_test, y_test = select_by_indices(X, y, test_idx)

    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = "./saved_model"
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "features.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "target": args.target,
                "feature_columns": list(X.columns),
                "drop_columns": drop_cols,
                "split": {
                    "split_random": bool(args.split_random),
                    "exclude_q_min": args.exclude_q_min,
                    "exclude_q_max": args.exclude_q_max,
                    "exclude_alpha_min": args.exclude_alpha_min,
                    "exclude_alpha_max": args.exclude_alpha_max,
                    "train_fractions": train_fractions,
                },
            },
            f,
            indent=2,
        )

    model_builders = build_model_builders(args.seed)
    rng = np.random.default_rng(args.seed)
    # train_order = X_train.index.to_numpy()
    train_order = np.arange(len(X_train))
    val_order = np.arange(len(X_val))
    print("train∩val", len(X_train.index.intersection(X_val.index)))
    print("train∩test", len(X_train.index.intersection(X_test.index)))
    print("val∩test", len(X_val.index.intersection(X_test.index)))
    if args.shuffle_fractions:
        rng.shuffle(train_order)
        rng_val = np.random.default_rng(args.seed + 1)
        rng_val.shuffle(val_order)
    metrics = []
    plots_by_fraction = {frac: [] for frac in train_fractions}
    shap_plots_by_fraction = {frac: [] for frac in train_fractions}
    feature_plots_by_fraction = {frac: {} for frac in train_fractions}
    for name in model_names:
        print(f"Training model: {name}")
        builder = model_builders[name]
        for frac in train_fractions:
            X_train_sub, y_train_sub = get_fraction_subset(X_train, y_train, train_order, frac)
            X_val_sub, y_val_sub = get_fraction_subset(X_val, y_val, val_order, frac)
            model = builder()
            model.fit(X_train_sub, y_train_sub)

            frac_tag = int(round(frac * 100))
            model_path = os.path.join(model_dir, f"{name}_train{frac_tag:03d}.joblib")
            joblib.dump(model, model_path)
            print(f"Saved model: {model_path}")

            if args.shap:
                shap_values, shap_X, _ = compute_shap_values(
                    model,
                    X_train_sub,
                    list(X.columns),
                    seed=args.seed,
                    max_samples=args.shap_samples,
                )
                if shap_values is not None and shap_X is not None:
                    shap_plots_by_fraction[frac].append(
                        {"label": name, "shap_values": shap_values, "X": shap_X}
                    )
            val_preds = model.predict(X_val_sub)
            test_preds = model.predict(X_test)
            val_metrics = evaluate_preds(y_val_sub, val_preds)
            test_metrics = evaluate_preds(y_test, test_preds)

            feature_plots_by_fraction[frac][name] = test_preds

            metrics.append(
                {
                    "model": name,
                    "train_fraction": frac,
                    "n_train": len(X_train_sub),
                    "test_mse": test_metrics["mse"],
                    "test_r2": test_metrics["r2"],
                    "test_mae": test_metrics["mae"],
                    "test_max_abs_error": test_metrics["max_abs_error"],
                }
            )

            plots_by_fraction[frac].append(
                {
                    "label": name,
                    "y_true": y_test,
                    "y_pred": test_preds,
                }
            )

    metrics_df = pd.DataFrame(metrics).sort_values(by=["train_fraction", "test_mse"], ascending=[True, True])
    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    for frac in train_fractions:
        if args.shap:
            save_shap_summary_grid(shap_plots_by_fraction[frac], args.output_dir, frac)
        plots = plots_by_fraction.get(frac, [])
        if not plots:
            continue
        frac_tag = int(round(frac * 100))
        # title = f"Test set (train {frac_tag}%)"
        title = None
        if len(plots) <= 4:
            plot_path = os.path.join(
                args.output_dir,
                f"test_scatter_train{frac_tag:03d}_grid.pdf",
            )
            save_scatter_grid(plots, plot_path, title)
        else:
            for chunk_idx in range(0, len(plots), 4):
                chunk = plots[chunk_idx : chunk_idx + 4]
                grid_tag = (chunk_idx // 4) + 1
                plot_path = os.path.join(
                    args.output_dir,
                    f"test_scatter_train{frac_tag:03d}_grid{grid_tag:02d}.pdf",
                )
                save_scatter_grid(chunk, plot_path, title)

    if sweep_feature:
        sweep_df = run_r2_sweep(
            X,
            y,
            sweep_feature,
            model_builders,
            model_names,
            train_fractions,
            args.seed,
            args.output_dir,
            args.r2_bins,
            shuffle_fractions=args.shuffle_fractions,
        )
        if not sweep_df.empty:
            sweep_path = os.path.join(
                args.output_dir, f"r2_vs_{sweep_feature.lower()}.csv"
            )
            sweep_df.to_csv(sweep_path, index=False)

    print("Models trained:", ", ".join(metrics_df["model"].tolist()))
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
