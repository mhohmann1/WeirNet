import numpy as np


a = np.load("Data/PKW_Efficiency_Dataset/pc/part_00001.npz", allow_pickle=True)
print(a["original_pcd"].shape)