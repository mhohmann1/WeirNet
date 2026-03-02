import numpy as np
import pandas as pd
import os
from args import args
import torch
from torch.utils.data import Dataset, random_split
from utils import random_pc
from augmentor import translate_pointcloud, jitter_pointcloud
import trimesh
from torch_geometric.data import Data

class RowDataset(Dataset):
    def __init__(self, npz_path, point_path, augmentation=False):
        # Load the .npz
        loaded = np.load(npz_path, allow_pickle=True)
        data = loaded['data']
        columns = loaded['columns']

        # Restore as DataFrame for convenience
        self.df = pd.DataFrame(data, columns=columns)
        self.path_to_pointclouds = point_path

        self.augmentation = augmentation
        self._filter_missing_pointclouds()
        self._build_numeric_df()

    @staticmethod
    def _model_to_name(value):
        if isinstance(value, str):
            return value
        return str(int(value)).zfill(5)

    @staticmethod
    def _model_to_number(value):
        if isinstance(value, str):
            digits = "".join(ch for ch in value if ch.isdigit())
            return float(digits) if digits else 0.0
        return float(value)

    def _build_numeric_df(self):
        self.df_numeric = self.df.copy()
        if "Modell" in self.df_numeric.columns:
            self.df_numeric["Modell"] = self.df_numeric["Modell"].apply(self._model_to_number)
        self.df_numeric = self.df_numeric.apply(pd.to_numeric, errors="coerce")

    def _filter_missing_pointclouds(self):
        if not os.path.isdir(self.path_to_pointclouds):
            return
        available = set()
        for fname in os.listdir(self.path_to_pointclouds):
            if fname.startswith("part_") and fname.endswith(".npz"):
                available.add(fname[len("part_"):-len(".npz")])
        if not available or "Modell" not in self.df.columns:
            return
        model_names = self.df["Modell"].apply(self._model_to_name)
        mask = model_names.isin(available)
        if not mask.all():
            self.df = self.df.loc[mask].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def split(self, train_ratio=0.8, val_ratio=0.1):
        num_samples = len(self)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        train_end = int(train_ratio * num_samples)
        val_end = train_end + int(val_ratio * num_samples)
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        return train_indices, val_indices, test_indices


    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name_model_value = self._model_to_name(row["Modell"])

        path_pointcloud = os.path.join(self.path_to_pointclouds, f"part_{name_model_value}.npz")
        pc_loaded = np.load(path_pointcloud, allow_pickle=True)
        pc_array = pc_loaded['normalized_pcd']
        pc_array = random_pc(pc_array, num_points=args.n_points)
        if self.augmentation:
            pc_array = translate_pointcloud(pc_array)
            pc_array = jitter_pointcloud(pc_array)
        pc_loaded = torch.tensor(pc_array, dtype=torch.float32)

        # Convert to tensor (optional: handle numeric/categorical separately)
        row_tensor = torch.tensor(self.df_numeric.iloc[idx].values, dtype=torch.float32)
        return row_tensor, pc_loaded

class GNNDataset(Dataset):
    def __init__(self, npz_path, stl_path, normalize=False):
        # Load the .npz
        loaded = np.load(npz_path, allow_pickle=True)
        data = loaded['data']
        columns = loaded['columns']

        # Restore as DataFrame for convenience
        self.df = pd.DataFrame(data, columns=columns)
        self.path_to_meshes = stl_path

        self.normalize = normalize
        self._filter_missing_meshes()
        self._build_numeric_df()

    @staticmethod
    def _model_to_name(value):
        if isinstance(value, str):
            return value
        return str(int(value)).zfill(5)

    @staticmethod
    def _model_to_number(value):
        if isinstance(value, str):
            digits = "".join(ch for ch in value if ch.isdigit())
            return float(digits) if digits else 0.0
        return float(value)

    def _build_numeric_df(self):
        self.df_numeric = self.df.copy()
        if "Modell" in self.df_numeric.columns:
            self.df_numeric["Modell"] = self.df_numeric["Modell"].apply(self._model_to_number)
        self.df_numeric = self.df_numeric.apply(pd.to_numeric, errors="coerce")

    def _filter_missing_meshes(self):
        if not os.path.isdir(self.path_to_meshes):
            return
        available = set()
        for fname in os.listdir(self.path_to_meshes):
            if fname.endswith(".stl"):
                base = fname[:-len(".stl")]
                if base.startswith("part_"):
                    base = base[len("part_"):]
                available.add(base)
        if not available or "Modell" not in self.df.columns:
            return
        model_names = self.df["Modell"].apply(self._model_to_name)
        mask = model_names.isin(available)
        if not mask.all():
            self.df = self.df.loc[mask].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def split(self, train_ratio=0.8, val_ratio=0.1):
        num_samples = len(self)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        train_end = int(train_ratio * num_samples)
        val_end = train_end + int(val_ratio * num_samples)
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        return train_indices, val_indices, test_indices

    def min_max_normalize(self, data):
        min_vals, _ = data.min(dim=0, keepdim=True)
        max_vals, _ = data.max(dim=0, keepdim=True)
        normalized_data = (data - min_vals) / (max_vals - min_vals)
        return normalized_data

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name_model_value = self._model_to_name(row["Modell"])

        path_stl = os.path.join(self.path_to_meshes, f"part_{name_model_value}.stl")
        if not os.path.isfile(path_stl):
            path_stl = os.path.join(self.path_to_meshes, f"{name_model_value}.stl")
        mesh = trimesh.load_mesh(path_stl, force="mesh")
        edge_index = torch.tensor(np.array(mesh.edges).T, dtype=torch.long)
        x = torch.tensor(mesh.vertices, dtype=torch.float)

        if self.normalize:
            x = self.min_max_normalize(x)

        y = torch.tensor(row.values[-1], dtype=torch.float)

        mesh = Data(x=x, edge_index=edge_index, y=y)
        row_tensor = torch.tensor(self.df_numeric.iloc[idx].values, dtype=torch.float32)
        return row_tensor, mesh
