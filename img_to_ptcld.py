

import numpy as np
import torch
import open3d as o3d

observations = torch.load("train_observations_lst.pt")

step_index = 3#8 #3 #8 #15 #8 #3
rgb_image = observations[step_index]['rgb']#[:, 80:-80]
depth_image = observations[step_index]['depth']#[:, 80:-80]
semantic_image = observations[step_index]['semantic']

visualizer = o3d.visualization.Visualizer()

# ================================ image ==========================

# Add image to visualizer
visualizer.create_window()
image = o3d.geometry.Image(np.array(rgb_image))
visualizer.add_geometry(image)

# Set camera view
visualizer.get_render_option().background_color = np.asarray([0, 0, 0])
visualizer.get_view_control().set_zoom(0.5)
# visualizer.run()


# =================================== rgb depth ===================

#reate depth map
depth_map = o3d.geometry.Image(np.array(depth_image))

# Create point cloud from depth map
intrinsic = o3d.camera.PinholeCameraIntrinsic(width=640, height=480, fx=500, fy=500, cx=320, cy=240)
extrinsic = np.eye(4)
pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_map, intrinsic, extrinsic)
# Create RGBD image from color and depth images
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    o3d.geometry.Image(np.array(rgb_image)),
    o3d.geometry.Image(np.array(depth_image)),
depth_scale=1,
convert_rgb_to_intensity=False
)
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(rgbd_image)
#vis.run()

# Create point cloud from RGBD image
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    o3d.camera.PinholeCameraIntrinsic(
        width=640, height=480, fx=500, fy=500, cx=320, cy=240
    )
)

# Convert point cloud to PyTorch tensor
points = np.array(pcd.points)
point_cloud = torch.from_numpy(points).float()

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)
#vis.run()

#===================================== semantic =======================

import pandas as pd
df = pd.read_csv('matterport_category_mappings.tsv', sep='  +')
df['nyuClass'].unique().tolist().index('watch')

import seaborn as sns
classes = df['nyuClass'].unique()
n_colors = len(classes)
color_palette = sns.color_palette("husl", n_colors)
palette_rgb = []
for color in color_palette:
    color_rgb = tuple(int(x * 255) for x in color)
    palette_rgb.append(color_rgb) 
class_color_dict = dict(zip(classes, palette_rgb))
df['nyuColor'] = df['nyuClass'].map(class_color_dict)
class_color_index_dict = dict(zip(classes, np.arange(len(classes))))
df['nyuColorIndex'] = df['nyuClass'].map(class_color_index_dict)


semantic_image = semantic_image.squeeze()
def int_to_nyuColor_Rgb(x):
    point = df['nyuColor'][df['index'] == x].values
    print(type(point))
    if len(point) > 0:
        return np.array(point[0])
sem_img_lst = []
class_lst = []
for x in semantic_image:
    sem_img_sub_lst = []
    for y in x:
        point = df['nyuColor'][df['index'] == y].values
        class_obj = df['nyuClass'][df['index'] == y].values
        if len(class_obj) > 0:
            class_lst.append(class_obj[0])
        import ipdb; ipdb.set_trace()
        if len(point) > 0:
            sem_img_sub_lst.append(point[0])
        else:
            sem_img_sub_lst.append(np.array([200,200,200], dtype=np.uint8))
    # TODO -- unequal arrs this will crash 
    try:
        sem_img_sub_arr = np.array(sem_img_sub_lst, dtype=np.uint8)
    except ValueError:
        import ipdb; ipdb.set_trace()
    sem_img_lst.append(sem_img_sub_arr)
sem_img_arr = np.array(sem_img_lst, dtype=np.uint8)
#semantic_point_cloud = np.vectorize(int_to_nyuColor_Rgb)(semantic_image)
#mask = np.logical_not(np.equal(semeantic_point_cloud, None))
#semantic_point_cloud = semantic_point_cloud.compress(mask)
print(list(set(class_lst)))

import colorsys
def make_rgb_palette(n=40):
    HSV_tuples = [(x*1.0/n, 0.8, 0.8) for x in range(n)]
    RGB_map = np.array(list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)))
    return RGB_map

colors = make_rgb_palette(45)
semantic_colors = colors[semantic_image % 45] * 255
semantic_colors = semantic_colors.astype(np.uint8)

# Create RGBD image from color and depth images
semantic_depth_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    o3d.geometry.Image(np.array(sem_img_arr)),
    o3d.geometry.Image(np.array(depth_image)),
depth_scale=1,
convert_rgb_to_intensity=False
)
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(semantic_depth_image)
vis.run()

# Create point cloud from RGBD image
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    semantic_depth_image,
    o3d.camera.PinholeCameraIntrinsic(
        width=640, height=480, fx=500, fy=500, cx=320, cy=240
    )
)

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)
vis.run()

"""
# Extract point cloud data
fx = 525.0
fy = 525.0
cx = 319.5
cy = 239.5

depth_scale = 1000.0
depth_intrinsics = torch.tensor([[fx, 0, cx],
                                 [0, fy, cy],
                                 [0, 0, 1]])

depth_image = torch.tensor(depth_image.astype('float32'))
depth_image /= depth_scale
depth_image[depth_image == 0] = float('nan')

x, y = torch.meshgrid(torch.arange(0, depth_image.shape[0]),
                      torch.arange(0, depth_image.shape[1]))
z = depth_image
points = torch.stack([x * z, y * z, z], dim=-1)
points = points.reshape(-1, 3)
points = torch.matmul(points, torch.inverse(depth_intrinsics).t())

# Convert point cloud data to PyTorch tensor
points = torch.tensor(points)
"""

