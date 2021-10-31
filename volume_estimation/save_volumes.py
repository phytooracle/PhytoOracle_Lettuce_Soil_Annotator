import numpy as np
import argparse
import os
import pandas as pd
import glob
import open3d as o3d
# Functions
# -----------------------------------------------------------------------------------------------------------

def save_volumes(indir, csv_name):

    df = pd.DataFrame(columns = ['plant_name', 'date', 'segmented_convex_hull_volume'])

    plant_dirs = os.path.join(indir, '*')

    for plant_dir in plant_dirs:
        try:
            plant_name = os.path.basename(plant_dir)

            pcd_path = os.path.join(plant_dir, 'combined_unregistered_plant.ply')
    
            pcd = o3d.io.read_point_cloud(pcd_path)


            # store bounding volume in csv
            hull,_ = pcd.compute_convex_hull()

            hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
            hull_ls.paint_uniform_color((1, 0, 0))
            # o3d.visualization.draw_geometries([pcd2, hull_ls])


            hull_volume = hull.get_volume()

            pcd_measurements = [plant_name, date, hull_volume]

            a_series = pd.Series(pcd_measurements, index = df.columns)
            df = df.append(a_series, ignore_index=True)
        except:
            print('No pcd')

    df.to_csv(os.path.join(indir, csv_name +  '.csv'))


# -----------------------------------------------------------------------------------------------------------

# if not os.path.isdir('./data'):
#     raise Exception("./data dir does not exist. You should mount data to this directory.")

    

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--indir', type = str)
parser.add_argument('-f', '--vcsv_name', default = 'hull_volumes', type = str)


args = parser.parse_args()

save_volumes(args.indir,  args.vcsv_name)
