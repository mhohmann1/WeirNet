import numpy as np

def translate_pointcloud(pointcloud, translation_range=(2./3., 3./2.)):
    xyz1 = np.random.uniform(low=translation_range[0], high=translation_range[1], size=[3])
    xyz2 = np.random.uniform(low=-10, high=10, size=[3])
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype("float32")
    return translated_pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud