from utils import knn
from torch.utils import data
from dataset import LettucePointCloudDataset
import torch
import numpy as np
import multiprocessing as mp
import time
import argparse
from tqdm import tqdm
import os
from dgcnn import DGCNN
import pandas as pd
import glob
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# Functions
# -----------------------------------------------------------------------------------------------------------




def generate_rotating_gif(array, gif_save_path, n_points=None, force_overwrite=False, scan_number=None):

    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111, projection='3d')
    x = array[:,0]
    y = array[:,1]
    z = array[:,2]
    c = array[:,3]
    cmap = 'Greens'
    ax.scatter(x, y, z,
               zdir='z',
               c = c,
               cmap = 'Dark2_r',
               marker='.',
               s=1,
    )
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.grid(False)
    ax.xaxis.pane.fill = False # Left pane
    ax.yaxis.pane.fill = False # Right pane
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    # Transparent panes
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # No ticks
    ax.set_xticks([]) 
    ax.set_yticks([]) 
    ax.set_zticks([])
    ax.set_box_aspect([max(x)-min(x),max(y)-min(y),max(z)-min(z)])
    def rotate(angle):
        ax.view_init(azim=angle)
    #rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 361, 2), interval=30)
    rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 361, 15), interval=300)
    #rot_animation.save('rotation.gif', dpi=80, writer='imagemagick')
    rot_animation.save(gif_save_path, dpi=80)

# -----------------------------------------------------------------------------------------------------------
    

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--indir', type = str)
parser.add_argument('--model_path', type = str)
parser.add_argument('-s', '--seg_dir', default = 'segmented_plants', type = str)
parser.add_argument('-f', '--vcsv_name', default = 'hull_volumes', type = str)
parser.add_argument('--K', type=int, default=51)
parser.add_argument('--n_samples', type=int, default=1500)

args = parser.parse_args()


# --------------------------------------------------------------------------------------------------------------

dataset = LettucePointCloudDataset(files_dir=args.indir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

model = DGCNN(num_classes=2).to(device)

model.load_state_dict(torch.load(args.model_path, map_location=device))

print(f'Model: {type(model).__name__}\n{"-"*15}')
model.eval()

print("Segmenting PointClouds...")
for (f_path, points) in tqdm(dataset):
    sampled_indices = np.random.choice(points.shape[0], args.n_samples, replace=False)
    sampled_indices_lookup = set(sampled_indices)
    sampled_points = points[sampled_indices]
    sampled_labels = model(torch.from_numpy(sampled_points).float().unsqueeze(0).to(device)).argmax(1).squeeze().cpu().numpy()
    
    labels = np.zeros(points.shape[0], dtype=int)
    labels[sampled_indices] = sampled_labels

    p = mp.Pool(mp.cpu_count() - 2)

    # start_time = time.time()
    other_labels = p.starmap(knn, [(points[i], sampled_points, sampled_labels, args.K) for i in range(points.shape[0]) if i not in sampled_indices_lookup])
    # print('KNN Log ===> Shape:', points.shape, '----', 'Time:', time.time()-start_time)

    others_indices_mask = np.ones(points.shape[0], dtype=bool)
    others_indices_mask[sampled_indices] = False

    labels[others_indices_mask] = other_labels

    np.save(f_path.replace('.ply', '.npy'), labels)

        # print(lables[0])
    # dropping no plant points
    arr = np.asarray(points)

    plant_arr = np.delete(arr, np.where( labels == 1), 0)
    plant_pcd = o3d.geometry.PointCloud()
    plant_pcd.points = o3d.utility.Vector3dVector(plant_arr)
    o3d.io.write_point_cloud(os.path.join(f_path.replace('.ply', '_plant.ply')), plant_pcd)

    arr = np.asarray(points)
    soil_arr = np.delete(arr, np.where( labels == 0), 0)
    soil_pcd = o3d.geometry.PointCloud()
    soil_pcd.points = o3d.utility.Vector3dVector(soil_arr)
    o3d.io.write_point_cloud(os.path.join(f_path.replace('.ply', '_soil.ply')), soil_pcd)





    # Adding in gif generation
    # labels == what array_path was loading
    # arr = np.asarray(points) == pcd_path was loading

    # args.indir = ./combined_pointclouds/a_plant_name/

    indir_filepath = args.indir
    plant_name = indir_filepath.split('/')[-1]
    gif_path = os.path.join('.', 'plant_reports', plant_name,  'combined_multiway_registered_soil_segmentation.gif')


    # pcd_array = generate_pointcloud_array_from_path(pcd_path, array_path)

    pcd_array = np.c_[arr, labels]

    generate_rotating_gif(pcd_array, gif_path)

