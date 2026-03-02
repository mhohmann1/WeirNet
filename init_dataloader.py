from args import args
from dataloader import RowDataset, GNNDataset
import torch
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
try:
    from torch_geometric.loader import DataLoader as GeoDataLoader
except Exception:
    GeoDataLoader = None

def _get_q_values(dataset):
    df = getattr(dataset, "df_numeric", None)
    if df is None:
        raise ValueError("Dataset has no numeric DataFrame; cannot locate Q values.")
    if "Q" in df.columns:
        return df["Q"].to_numpy()
    if df.shape[1] > 1:
        return df.iloc[:, 1].to_numpy()
    raise ValueError("Q column not found; expected column name 'Q' or index 1.")

def _split_by_q_range(dataset, q_min, q_max, train_ratio=0.8, val_ratio=0.1):
    q_vals = _get_q_values(dataset)
    q_min_val = -np.inf if q_min is None else q_min
    q_max_val = np.inf if q_max is None else q_max
    ood_mask = (q_vals >= q_min_val) & (q_vals <= q_max_val)

    ood_indices = np.nonzero(ood_mask)[0]
    id_indices = np.nonzero(~ood_mask)[0]

    if len(ood_indices) == 0:
        raise ValueError(f"exclude_q range produced 0 samples (min={q_min}, max={q_max}).")
    if len(id_indices) == 0:
        raise ValueError("exclude_q range covers entire dataset; no in-distribution samples left.")

    if len(id_indices) == 1:
        return id_indices, id_indices, ood_indices

    rng = np.random.default_rng(getattr(args, "seed", None))
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

    return train_indices, val_indices, ood_indices

def _subset_by_fraction(dataset, fraction, seed):
    if fraction is None or fraction >= 1.0:
        return dataset
    if fraction <= 0.0:
        raise ValueError(f"train_fraction must be in (0, 1], got {fraction}.")
    n_total = len(dataset)
    n_keep = max(1, int(n_total * fraction))
    if n_keep >= n_total:
        return dataset
    rng = np.random.default_rng(seed)
    indices = np.arange(n_total)
    rng.shuffle(indices)
    subset_indices = np.sort(indices[:n_keep])
    return Subset(dataset, subset_indices)

def init_dataset(npz_path, pc_dir, stl_dir, augmentation, split_random=False):
    if args.model == "GNN":
        dataset = GNNDataset(npz_path=npz_path, stl_path=stl_dir, normalize=augmentation)
    else:
        dataset = RowDataset(npz_path=npz_path, point_path=pc_dir, augmentation=augmentation)

    use_q_exclusion = any([
        getattr(args, "exclude_q_min", None) is not None,
        getattr(args, "exclude_q_max", None) is not None,
    ])

    if use_q_exclusion:
        train_indices, val_indices, test_indices = _split_by_q_range(
            dataset, args.exclude_q_min, args.exclude_q_max
        )
        train_data = Subset(dataset, train_indices)
        valid_data = Subset(dataset, val_indices)
        test_data = Subset(dataset, test_indices)
    elif split_random:
        train_size = int(0.8 * len(dataset))
        valid_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - valid_size

        train_data, valid_data, test_data = random_split(dataset, [train_size, valid_size, test_size])

    else:
        train_indices = np.loadtxt("train_val_test/train_indices.txt", dtype=int)
        val_indices = np.loadtxt("train_val_test/val_indices.txt", dtype=int)
        test_indices = np.loadtxt("train_val_test/test_indices.txt", dtype=int)

        train_data = Subset(dataset, train_indices)
        valid_data = Subset(dataset, val_indices)
        test_data = Subset(dataset, test_indices)

    train_data = _subset_by_fraction(train_data, getattr(args, "train_fraction", 1.0), getattr(args, "seed", None))

    if args.model == "GNN":
        if GeoDataLoader is None:
            raise RuntimeError("torch_geometric is required for GNN DataLoader")
        loader_cls = GeoDataLoader
    else:
        loader_cls = DataLoader

    train_loader = loader_cls(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)
    valid_loader = loader_cls(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True)
    test_loader = loader_cls(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True)

    return train_loader, valid_loader, test_loader
