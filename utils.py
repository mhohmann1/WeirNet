import numpy as np
import torch

def random_pc(data, num_points=2048):
    idx = np.random.choice(data.shape[0], num_points, replace=False)
    data = data[idx, :]
    return data

def r2_score(output, target):
    """Compute R-squared score."""
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2
