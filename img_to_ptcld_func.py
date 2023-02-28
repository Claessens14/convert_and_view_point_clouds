


# ================================ semantic-depth image to point cloud function  ==============


import numpy as np
import torch
import open3d as o3d
import colorsys

observations = torch.load("train_observations_lst.pt")

step_index = 3#8 #3 #8 #15 #8 #3
rgb_image = observations[step_index]['rgb']#[:, 80:-80]
depth_image = observations[step_index]['depth']#[:, 80:-80]
semantic_image = observations[step_index]['semantic']


def img_to_ptcld(semantic_image, depth_image):
    '''
    Take in semantic map and depth information that is associated with a 
    single observation
    return: point cloud
    '''
    
    def make_rgb_palette(n=40):
        HSV_tuples = [(x*1.0/n, 0.8, 0.8) for x in range(n)]
        RGB_map = np.array(list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)))
        return RGB_map

    colors = make_rgb_palette(45)
    semantic_colors = colors[semantic_image % 45] * 255
    semantic_colors = semantic_colors.astype(np.uint8) #(480, 640, 1, 3)
    # Create RGBD image from color and depth images
    semantic_depth_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.array(semantic_colors)),
        o3d.geometry.Image(np.array(depth_image)),
    depth_scale=1,
    convert_rgb_to_intensity=False
    )
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(semantic_depth_image)
    #vis.run()
    
    # Create point cloud from RGBD image
    import ipdb; ipdb.set_trace()
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

img_to_ptcld(semantic_image, depth_image)




