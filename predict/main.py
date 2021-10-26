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


# Functions
# -----------------------------------------------------------------------------------------------------------

def separate_plant_points_and_save_volumes(indir, outdir, csv_name):

    seg_dir = os.path.join(indir, outdir)

    try:
        os.mkdir(seg_dir)
    except:
        pass


    df = pd.DataFrame(columns = ['plant_name', 'date', 'segmented_convex_hull_volume']);df


    # def segment_and_calculate_volume(indir):

    full_plants = glob.glob(os.path.join(indir, '*.ply'))

    print(len(full_plants))

    for pcd_path in full_plants:
        # Visualize a pcd

        plant_name = os.path.basename(pcd_path).split('cropped')[0][:-1]
        date = os.path.basename(pcd_path).split('cropped')[1][1:].replace('.ply', '');date
        pcd = o3d.io.read_point_cloud(pcd_path)

        filename = os.path.basename(pcd_path)
        # o3d.visualization.draw_geometries([pcd])

        # corrisponding label
        lables = np.load(pcd_path.replace('.ply', '.npy'))


        # print(lables[0])
        # dropping no plant points
        arr = np.asarray(pcd.points)

        new_arr = np.delete(arr, np.where( lables == 0), 0);new_arr


        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(new_arr)

        o3d.io.write_point_cloud(os.path.join(seg_dir, filename.replace('.ply', '_segmented.ply')), pcd2)


        # store bounding volume in csv
        hull,_ = pcd2.compute_convex_hull()

        hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
        hull_ls.paint_uniform_color((1, 0, 0))
        # o3d.visualization.draw_geometries([pcd2, hull_ls])


        hull_volume = hull.get_volume()

        pcd_measurements = [plant_name, date, hull_volume]

        a_series = pd.Series(pcd_measurements, index = df.columns)
        df = df.append(a_series, ignore_index=True)

    df.to_csv(os.path.join(seg_dir, csv_name +  '.csv'))


# -----------------------------------------------------------------------------------------------------------

# if not os.path.isdir('./data'):
#     raise Exception("./data dir does not exist. You should mount data to this directory.")

    

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--indir', type = str)
parser.add_argument('--model_path', type = str)
parser.add_argument('-s', '--seg_dir', default = 'segmented_plants', type = str)
parser.add_argument('-f', '--vcsv_name', default = 'hull_volumes', type = str)
parser.add_argument('--K', type=int, default=51)
parser.add_argument('--n_samples', type=int, default=1500)

args = parser.parse_args()

# if args.use_given_model:
#     if not os.path.isfile('./new_trained_model/DGCNN.pth'):
#         raise Exception('./new_trained_model/DGCNN.pth is not a file. You have to mount your trained model when you use use_given_model argument.')

dataset = LettucePointCloudDataset(files_dir=args.indir)

device = torch.device("cuda:0")
print(f'Device: {device}')

model = DGCNN(num_classes=2).to(device)
# if args.use_given_model:
#     model.load_state_dict(torch.load('./new_trained_model/DGCNN.pth', map_location=device))
# else:
#     model.load_state_dict(torch.load('./pretrained_model/DGCNN.pth', map_location=device))

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

