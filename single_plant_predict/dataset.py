
from numpy.linalg.linalg import norm
from utils import normalize_points
from torch.utils.data import Dataset
import os
import open3d as o3d
import numpy as np
from utils import normalize_points, reverse_normalization
import torch
import glob

# class LettucePointCloudDataset(Dataset):
#     def __init__(self, files_dir):
#         self.files = []
#         for f in os.listdir(files_dir):
#             if f.endswith('.ply'):
#                 self.files.append(os.path.join(files_dir, f))
    
#     def __len__(self):
#         return len(self.files)
    
#     def __getitem__(self, idx):
#         pcd = o3d.io.read_point_cloud(self.files[idx])
#         points = np.array(pcd.points)
#         points, 

#         return self.files[idx], normalize_points(points)
def LettucePointCloudDataset(files_dir):

    mods = []

    files = glob.glob(os.path.join(files_dir, '*.ply'))

    dataset = []

    for i in files:
        pcd = o3d.io.read_point_cloud(i)
        points = np.array(pcd.points)
        points, mod1, mod2 = normalize_points(points)
        mods.append([mod1, mod2])
        dataset.append((i, points))

    return dataset, mods

    