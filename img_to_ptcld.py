

import numpy as np
import torch
import open3d as o3d

observations = torch.load("train_observations_lst.pt")

print(observations[0].keys())
step_index = 8 #3
rgb_image = observations[step_index]['rgb'][:, 80:-80]
depth_image = observations[step_index]['depth'][:, 80:-80]



visualizer = o3d.visualization.Visualizer()

# Add image to visualizer
visualizer.create_window()
image = o3d.geometry.Image(np.array(rgb_image))
visualizer.add_geometry(image)

# Set camera view
visualizer.get_render_option().background_color = np.asarray([0, 0, 0])
visualizer.get_view_control().set_zoom(0.5)
visualizer.run()


#reate depth map
depth_map = o3d.geometry.Image(np.array(depth_image))

# Create point cloud from depth map
intrinsic = o3d.camera.PinholeCameraIntrinsic(width=640, height=480, fx=500, fy=500, cx=320, cy=240)
extrinsic = np.eye(4)
pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_map, intrinsic, extrinsic)

# Create RGBD image from color and depth images
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    o3d.geometry.Image(np.array(rgb_image)),
    o3d.geometry.Image(np.array(depth_image))
)

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
vis.run()






"""
# Extract point cloud data
fx = 525.0
fy = 525.0
cx = 319.5
cy = 239.5

depth_scale = 1000.0
import ipdb; ipdb.set_trace()
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

