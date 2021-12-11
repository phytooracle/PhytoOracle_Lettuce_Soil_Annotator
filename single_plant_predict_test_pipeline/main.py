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
import matplotlib

# Test Command
# /home/travis/data/season_10/combined_pointclouds/Pacific_74
# /home/travis/data/ml_weights/dgcnn_trained_on_normalized_arman_format_training_data_perfect_partial_full_30epcs_98acc_47loss.pth
# python3 main.py -i /home/travis/data/season_10/combined_pointclouds/Pacific_74 --model_path /home/travis/data/ml_weights/dgcnn_trained_on_normalized_arman_format_training_data_perfect_partial_full_30epcs_98acc_47loss.pth



# Functions
# -----------------------------------------------------------------------------------------------------------




def generate_rotating_gif(array, gif_save_path, n_points=None, force_overwrite=False, scan_number=None):

    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111, projection='3d')
    x = array[:,0]
    y = array[:,1]
    z = array[:,2]
    c = array[:,3]

    an_array = np.where(c == 1, 1, 2)

    colors = ['green','sienna']

    cmap_arr= matplotlib.colors.ListedColormap(colors)



    # cmap = 'Greens'
    ax.scatter(x, y, z,
               zdir='z',
               c = an_array,
               cmap = cmap_arr,
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
parser.add_argument('-p', '--plant_path', type = str)
parser.add_argument('-ef', '--expected_filename',default = 'foo',type = str)

parser.add_argument('--model_path', type = str)
parser.add_argument('-s', '--seg_dir', default = 'segmented_plants', type = str)
parser.add_argument('-f', '--vcsv_name', default = 'hull_volumes', type = str)
parser.add_argument('--K', type=int, default=51)
parser.add_argument('--n_samples', type=int, default=1500)

parser.add_argument('-po',
                    '--pointcloud_outdir',
                    help='Output directory for pointclouds',
                    default = 'segmentation_pointclouds',
                    metavar='str',
                    type=str)
                    

parser.add_argument('-fo',
                    '--figures_outdir',
                    help='Output directory for figures',
                    default = 'plant_reports',
                    metavar='str',
                    type=str)

# parser.add_argument('-pf',
#                     '--pointcloud_filename',
#                     help='Pointcloud Output filename',
#                     default = '',
#                     metavar='str',
#                     type=str)

args = parser.parse_args()


# --------------------------------------------------------------------------------------------------------------
# Prep


# Inputs
plant_path = args.plant_path
plant_name = os.path.basename(plant_path)
expected_filename = args.expected_filename

if expected_filename != 'foo':
    pcd_name = args.expected_filename
    pointcloud_indir = args.pointcloud_outdir
    full_pcd_input = os.path.join(pointcloud_indir, plant_name, pcd_name + '.ply')

else:
    pcd_name = 'combined_multiway_registered.ply'
    pointcloud_indir = plant_path
    full_pcd_input = os.path.join(pointcloud_indir, pcd_name)

# Outputs
pointcloud_outdir = args.pointcloud_outdir
figures_outdir = args.figures_outdir
plant_pointcloud_outdir = os.path.join(pointcloud_outdir, plant_name)




figures_outdir = args.figures_outdir
plant_figures_outdir = os.path.join(figures_outdir, plant_name)

full_soil_pcd_path = os.path.join(plant_pointcloud_outdir, 'segmentation_soil.ply')
full_plant_pcd_path = os.path.join(plant_pointcloud_outdir, 'segmentation_plant.ply')
full_label_npy_path = os.path.join(plant_pointcloud_outdir, 'soil_segmentation.npy')


full_csv_outpath = os.path.join(plant_figures_outdir, 'soil_segmentation.csv')
full_gif_outpath = os.path.join(plant_figures_outdir, 'soil_segmentation.gif')
    


# prepping output directory structure
if not os.path.exists(pointcloud_outdir):
    os.mkdir(pointcloud_outdir)

if not os.path.exists(plant_pointcloud_outdir):
    os.mkdir(plant_pointcloud_outdir)

if not os.path.exists(figures_outdir):
    os.mkdir(figures_outdir)

if not os.path.exists(plant_figures_outdir):
    os.mkdir(plant_figures_outdir)


# main
dataset = LettucePointCloudDataset(files_dir=full_pcd_input)

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

    np.save(full_label_npy_path, labels)

        # print(lables[0])
    # dropping no plant points
    arr = np.asarray(points)

    plant_arr = np.delete(arr, np.where( labels == 1), 0)
    plant_pcd = o3d.geometry.PointCloud()
    plant_pcd.points = o3d.utility.Vector3dVector(plant_arr)
    o3d.io.write_point_cloud(full_plant_pcd_path, plant_pcd)

    arr = np.asarray(points)
    soil_arr = np.delete(arr, np.where( labels == 0), 0)
    soil_pcd = o3d.geometry.PointCloud()
    soil_pcd.points = o3d.utility.Vector3dVector(soil_arr)
    o3d.io.write_point_cloud(full_soil_pcd_path, soil_pcd)

    # Adding in gif generation
    pcd_array = np.c_[arr, labels]

    generate_rotating_gif(pcd_array, full_gif_outpath)

