import numpy as np
import argparse
import os
import pandas as pd
import glob
import open3d as o3d
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy
from gtda.plotting import plot_point_cloud

# Functions
# -----------------------------------------------------------------------------------------------------------
def open_pcd(pcd_path):

    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd.estimate_normals()
    pcd.normalize_normals()
    
    return pcd

def downsample_pcd(pcd, voxel_size=0.05):

    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    return down_pcd


def visualize_pcd(pcd, extra=None):
    if extra:
        o3d.visualization.draw_geometries([pcd, extra])
    else:    
        o3d.visualization.draw_geometries([pcd])


def calculate_convex_hull_volume(pcd):
    hull, _ = pcd.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_vol = hull.get_volume()

    return hull_vol


def calculate_oriented_bb_volume(pcd):

    obb_vol = pcd.get_oriented_bounding_box().volume()

    return obb_vol


def calculate_axis_aligned_bb_volume(pcd):

    abb_vol = pcd.get_axis_aligned_bounding_box().volume()

    return abb_vol


def convert_point_cloud_to_array(pcd):
    # Convert point cloud to a Numpy array
    pcd_array = np.asarray(pcd.points)
    pcd_array  = pcd_array.astype(float)

    return pcd_array


def calculate_persistance_diagram(pcd_array):
    # Calculate persistance diagram
    VR = VietorisRipsPersistence(metric='euclidean', homology_dimensions=[0, 1, 2])  # Parameter explained in the text
    diagrams = VR.fit_transform(pcd_array[None, :, :])

    # Calculate the entropy
    PE = PersistenceEntropy()
    features = PE.fit_transform(diagrams)

    return features


def separate_features(features):

    zero = features[0][0]
    one = features[0][1]
    two = features[0][2]

    return zero, one, two

def get_min_max(pcd):
    
    max_x, max_y, max_z = pcd.get_max_bound()
    min_x, min_y, min_z = pcd.get_min_bound()

    return max_x, max_y, max_z, min_x, min_y, min_z

def save_volumes(indir, csv_name):

    df = pd.DataFrame(columns = ['plant_name', 'plant_convex_hull_volume', 'plant_oriented_bounding', 'plant_axis_aligned_bounding', 'max_x', 'max_y', 'max_z', 'min_x', 'min_y', 'min_z']#, 'persistence entropies_feature_0', 'persistence entropies_feature_1', 'persistence entropies_feature_2'])

    plant_dirs = glob.glob(os.path.join(indir, '*'))

    for plant_dir in plant_dirs:
        try:
            plant_name = os.path.basename(plant_dir)

            #pcd_path = os.path.join(plant_dir, 'combined_unregistered_plant.ply')
            pcd_path = os.path.join(plant_dir, 'combined_multiway_registered_plant.ply')
    
            pcd = open_pcd(pcd_path)

            print('Calculating hull volume.')
            hull_volume = calculate_convex_hull_volume(pcd)
            print('Calculating oriented bounding box volume.')
            obb = calculate_oriented_bb_volume(pcd)
            print('Calculating axis aligned bounding box volume.')
            abb = calculate_axis_aligned_bb_volume(pcd)
                      
            max_x, max_y, max_z, min_x, min_y, min_z = get_min_max(pcd)
            
#             print('Calculating persistance diagram and entropy.')
#             down_pcd = downsample_pcd(pcd)
#             pcd_array = convert_point_cloud_to_array(down_pcd)
#             features = calculate_persistance_diagram(pcd_array)
#             zero, one, two = separate_features(features)

            pcd_measurements = [plant_name, hull_volume, obb, abb, max_x, max_y, max_z, min_x, min_y, min_z]#, zero, one, two]

            a_series = pd.Series(pcd_measurements, index = df.columns)
            df = df.append(a_series, ignore_index=True)
        except:
            print('No point cloud found, cannot calculate hull volume.')

    df.to_csv(os.path.join(indir, csv_name +  '.csv'))


# -----------------------------------------------------------------------------------------------------------

# if not os.path.isdir('./data'):
#     raise Exception("./data dir does not exist. You should mount data to this directory.")

    

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--indir', type = str)
parser.add_argument('-f', '--vcsv_name', default = 'hull_volumes', type = str)


args = parser.parse_args()

save_volumes(args.indir,  args.vcsv_name)
