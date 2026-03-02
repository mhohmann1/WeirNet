import math
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
})

dataset = np.load("./Data/PKW_Efficiency_Dataset/combined_data.npz", allow_pickle=True)

columns = dataset["columns"]
data = dataset["data"]


def _decode_column_name(name):
    if isinstance(name, bytes):
        return name.decode("utf-8", errors="replace")
    return str(name)


def save_continuous_histograms(column_names, output_path, bins=200):
    decoded_columns = [_decode_column_name(col) for col in columns]
    name_to_index = {name: idx for idx, name in enumerate(decoded_columns)}

    label_size = 14
    tick_size = 12

    ncols = 3
    nrows = math.ceil(len(column_names) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.ravel(axes)

    colors = ["#F58518", "#4C78A8", "#54A24B"]
    for idx, name in enumerate(column_names):
        if name not in name_to_index:
            raise ValueError(f"Column not found: {name}")
        values = data[:, name_to_index[name]]
        
        if "Alpha_deg" == name:
            name = r"$\alpha$ [°]"
        elif "C_d" == name:
            name = r"$c_D$ [-]"
        elif "Q" == name:
            name = r"$Q$ [$\mathrm{m}^3/\mathrm{s}$]"
        
        hist, bin_edges = np.histogram(values, bins=bins, density=False)
        ax = axes[idx]
        color = colors[idx % len(colors)]
        ax.stairs(hist, bin_edges, color=color)
        ax.fill_between(bin_edges[:-1], hist, 0, step="post", color=color, alpha=0.3)
        ax.set_xlabel(name, fontsize=label_size)
        if idx == 0:
            ax.set_ylabel("Density", fontsize=label_size)
        else:
            ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=tick_size)

    for idx in range(len(column_names), len(axes)):
        axes[idx].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)


def save_correlation_heatmap(
    output_path, exclude_columns=None, triangle=None
):
    decoded_columns = [_decode_column_name(col) for col in columns]
    if exclude_columns:
        keep_indices = [
            idx for idx, name in enumerate(decoded_columns) if name not in exclude_columns
        ]
    else:
        keep_indices = list(range(len(decoded_columns)))

    filtered_columns = [decoded_columns[idx] for idx in keep_indices]
    filtered_data = data[:, keep_indices]

    corr = np.corrcoef(filtered_data, rowvar=False)
    n = corr.shape[0]

    cmap = plt.get_cmap("coolwarm")
    if triangle in {"upper", "lower"}:
        corr = corr.copy()
        mask = np.zeros_like(corr, dtype=bool)
        if triangle == "upper":
            mask |= np.tril(np.ones_like(corr, dtype=bool), k=-1)
        elif triangle == "lower":
            mask |= np.triu(np.ones_like(corr, dtype=bool), k=1)
        corr[mask] = np.nan
        cmap = cmap.copy()
        cmap.set_bad(color="white")

    # Scale figure size with feature count, but keep within reasonable bounds.
    size = min(max(6, 0.4 * n), 24)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def _format_heatmap_label(name):
        if "C_d" == name:
            return r"$c_D$"
        if "Q" == name:
            return r"$Q$"
        if "B_i" == name:
            return r"$B_i$"
        if "B_o" == name:
            return r"$B_o$"
        if "B" == name:
            return r"$B$"
        if "Alpha_deg" == name:
            return r"$\alpha$"
        if "Ts_stern" == name:
            return r"$T_{s,2}$"
        if "Ts_dach" == name:
            return r"$T_{s,3}$"
        if "W_o_u" == name:
            return r"$W_{o,u}$"
        if "W_o_d" == name:
            return r"$W_{o,d}$"
        return name

    heatmap_labels = [_format_heatmap_label(name) for name in filtered_columns]

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(heatmap_labels)
    ax.set_yticklabels(heatmap_labels)
    ax.tick_params(axis="x", labelrotation=90)

    # Reduce label size for dense matrices.
    label_size = 12 if n <= 30 else 8
    ax.tick_params(axis="both", labelsize=label_size)

    # Annotate correlation values.
    annot_size = 10 if n <= 10 else (7 if n <= 20 else 5)
    for i in range(n):
        for j in range(n):
            value = corr[i, j]
            if np.isnan(value):
                continue
            text_color = "white" if abs(value) >= 0.6 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                    fontsize=annot_size, color=text_color)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)


def save_features_vs_cd_plots(output_path, exclude_columns=None, feature_on_x=False):
    decoded_columns = [_decode_column_name(col) for col in columns]
    name_to_index = {name: idx for idx, name in enumerate(decoded_columns)}
    if "C_d" not in name_to_index:
        raise ValueError("Column not found: C_d")

    exclude = set(exclude_columns or [])
    exclude.add("C_d")

    feature_names = [name for name in decoded_columns if name not in exclude]
    if not feature_names:
        raise ValueError("No features left to plot against C_d")

    ncols = 3
    nrows = math.ceil(len(feature_names) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.ravel(axes)

    cd_idx = name_to_index["C_d"]
    cd_values = data[:, cd_idx]

    for i, name in enumerate(feature_names):
        if feature_on_x:
            x = data[:, name_to_index[name]]
            y = cd_values
        else:
            x = cd_values
            y = data[:, name_to_index[name]]
        ax = axes[i]
        if "Q" == name:
            name = r"$Q$ [$\mathrm{m}^3/\mathrm{s}$]"
        elif "B_i" == name:
            name = r"$B_i$ [mm]"
        elif "B_o" == name:
            name = r"$B_o$ [mm]"
        elif "B" == name:
            name = r"$B$ [mm]"
        elif "Alpha_deg" == name:
	        name = r"$\alpha$ [°]"
        elif "Ts_stern" == name:
            name = r"$T_{s,2}$ [mm]"
        elif "Ts_dach" == name:
            name = r"$T_{s,3}$ [mm]"
        elif "W_o_u" == name:
            name = r"$W_{o,u}$ [mm]"
        elif "W_o_d" == name:
            name = r"$W_{o,d}$ [mm]"
        ax.scatter(x, y, s=8, alpha=0.5, color="#4C78A8", edgecolor="none")
        ax.set_xlabel(name if feature_on_x else r"$c_D$ [-]", fontsize=12)
        ax.set_ylabel(r"$c_D$ [-]" if feature_on_x else name, fontsize=12)
        ax.tick_params(axis="both", labelsize=10)

    for i in range(len(feature_names), len(axes)):
        axes[i].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)


n_features = data.shape[1]
ncols = 3
nrows = math.ceil(n_features / ncols)

label_size = 14
tick_size = 12

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows))
axes = np.ravel(axes)

for i in range(n_features):
    col_name = _decode_column_name(columns[i])
    axes[i].hist(data[:, i], bins=25, color="#4C78A8", edgecolor="white")
    axes[i].set_title(col_name, fontsize=label_size)
    axes[i].tick_params(axis="both", labelsize=tick_size)

for i in range(n_features, len(axes)):
    axes[i].axis("off")

plt.tight_layout()
plt.savefig("feature_histograms.png", dpi=300)

save_continuous_histograms(
    ["C_d", "Q", "Alpha_deg"],
    "histograms.png",
)

save_correlation_heatmap(
    "correlation_heatmap.png",
    exclude_columns={'Modell', 'Pressure', 'h_O', 'h_t', 'H_t', 'B_b' ,'N_B_i', 'N_B_o', 'W_i_u', 'W_i_d', 'Cycles', 'Alpha_rad', 'W' , 'Ht_P', 'T_s', 'L', 'W_u', 'P'},
    triangle="lower",
)

save_features_vs_cd_plots(
    "features_vs_c_d.png",
    exclude_columns={'Modell', 'Pressure', 'h_O', 'h_t', 'H_t', 'B_b' ,'N_B_i', 'N_B_o', 'W_i_u', 'W_i_d', 'Cycles', 'Alpha_rad', 'W' , 'Ht_P', 'T_s', 'L', 'W_u', 'P'},
)

save_features_vs_cd_plots(
    "features_vs_c_d_xy.png",
    exclude_columns={'Modell', 'Pressure', 'h_O', 'h_t', 'H_t', 'B_b' ,'N_B_i', 'N_B_o', 'W_i_u', 'W_i_d', 'Cycles', 'Alpha_rad', 'W' , 'Ht_P', 'T_s', 'L', 'W_u', 'P'},
    feature_on_x=True,
)

plt.show()
