import numpy as np, pandas as pd
import torch, os, open3d as o3d
import matplotlib.pyplot as plt

"""
This file is running with the hab-dev conda env. It's purpuse is to test point cloud conversion
for semantic depth images, AND to look at labelling annotations of the file. The .pt file it loads
has annotations integers with a mean of around 700. What do this corriposond to? This file hopes
to answer that question, further discover how the meta-labels can be applied, so this dataset
can be compared to other semantic depth datasets. RUN IN TMUX FOR COLOURS

March 6th, 2023, Jacob Claessens
"""

observations = torch.load("train_observations_lst.pt")
step_index = 3#8#3#8 #3 #8 #15 #8 #3
end = len(observations[step_index]['rgb'])
rgb_image = observations[step_index]['rgb'][:end]#[:, 80:-80]
depth_image = observations[step_index]['depth'][:end]#[:, 80:-80]
semantic_image = observations[step_index]['semantic'][:end]

#===================================== semantic =======================
tsv_name = "hm3dsem_category_mappings"; df = pd.read_csv(tsv_name+'.tsv', sep='\t')
#tsv_name = 'matterport_category_mappings'; df = pd.read_csv(tsv_name+'.tsv', sep='  +')
class_name = 'category'; filename = f"sem_img_arr-stepIdx_{step_index}_{class_name}_{tsv_name}.pt"
sem_img_arr = None; sem_img_lst = []; class_lst = []
def add_color_columns():
    import seaborn as sns
    classes = df[class_name].unique(); n_colors = len(classes)
    color_palette = sns.color_palette("husl", n_colors)
    palette_rgb = []
    for color in color_palette:
        color_rgb = tuple(int(x * 255) for x in color)
        palette_rgb.append(color_rgb) 
    class_color_dict = dict(zip(classes, palette_rgb))
    df[class_name + "_color"] = df[class_name].map(class_color_dict)
    class_color_index_dict = dict(zip(classes, np.arange(len(classes))))
    df[class_name+'_color_index'] = df[class_name].map(class_color_index_dict)
add_color_columns()
if os.path.exists(filename):
    obj = torch.load(filename)
    sem_img_arr = obj['sem_img_arr']
    class_lst = obj['class_lst']
    print("Model loaded successfully.")
else:
    print(f"File {filename} not found.")
    #df.to_excel('data.xlsx', index=False)
    #filtered_df = df[df['raw_category'].str.contains('wall')]
    #print(filtered_df)
    #exit()
    semantic_image = semantic_image.squeeze()
   # def int_to_nyuColor_Rgb(x):
   #     point = df[class_name+'_color'][df['index'] == x].values
   #     print(type(point))
   #     if len(point) > 0:
   #         return np.array(point[0])
    for x in semantic_image:
        sem_img_sub_lst = []
        for y in x:
            print(df.head()); exit()
            point = df[class_name + "_color"][df['index'] == y].values
            class_obj = df[class_name][df['index'] == y].values
            if len(class_obj) > 0:
                class_lst.append(class_obj[0])
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
    torch.save({'sem_img_arr':sem_img_arr, 'class_lst':class_lst}, filename)
#semantic_point_cloud = np.vectorize(int_to_nyuColor_Rgb)(semantic_image)
#mask = np.logical_not(np.equal(semeantic_point_cloud, None))
#semantic_point_cloud = semantic_point_cloud.compress(mask)
color_legend = {el:df[class_name + "_color"][df[class_name]==el].values[0] for el in class_lst} 
print(color_legend)
def colored_background(r, g, b, text):
    # https://stackoverflow.com/questions/70519979/printing-with-rgb-background
    return f"\033[48;2;{r};{g};{b}m{text}\033[0m"
for class_name in color_legend:
    rgb = color_legend[class_name]
    print(colored_background(rgb[0], rgb[1], rgb[2], class_name))
#o3d.visualization.gui.Label.text = 'hello'
#o3d.visualization.gui.Label("hello4")
#text = o3d.geometry.Text("Legend", 10)
#text.paint_uniform_color([1, 1, 1]) # set text color
# Add point cloud and text to viewport
#import colorsys
#def make_rgb_palette(n=40):
#    HSV_tuples = [(x*1.0/n, 0.8, 0.8) for x in range(n)]
#    RGB_map = np.array(list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)))
#    return RGB_map
#colors = make_rgb_palette(45)
#semantic_colors = colors[semantic_image % 45] * 255
#semantic_colors = semantic_colors.astype(np.uint8)

# Create RGBD image from color and depth images
semantic_depth_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    o3d.geometry.Image(np.array(sem_img_arr)),
    o3d.geometry.Image(np.array(depth_image)),
depth_scale=1,
convert_rgb_to_intensity=False
)
#vis = o3d.visualization.Visualizer()
#vis.create_window()
#vis.add_geometry(semantic_depth_image)
#vis.run()

# Create point cloud from RGBD image
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    semantic_depth_image,
    o3d.camera.PinholeCameraIntrinsic(
        width=640, height=480, fx=500, fy=500, cx=320, cy=240
    )
)


sem_img_flat_idx = semantic_image.reshape(640*end, 1)
sem_img_flat_col_arr = sem_img_arr.reshape(640*end, 1, 3)
points = np.asarray(pcd.points)
#C = np.array ( [ [255, 0, 01, [0, 255, 01, (0, 0, 255]])

#fig = plt.figure()
#ax = fig.add_subplot(111, projection = '3d')
#ax.scatter(points[:,0],points[:,1], points[:,2]) # c = sem_img_flat_col_arr/255.0)
#ax. scatter(points, c = sem_img_flat_arr/255.0)
#ax.scatter(points[:,0],points[:,1], points[:,2]), c = sem_img_flat_col_arr/255.0)
#for i in range(len(sem_img_flat_col_arr)):
#    ax.scatter(points[i,0],points[i,1], points[i,2], color = sem_img_flat_col_arr[i]/255.0)
#plt.show()

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)
vis.run()


"""
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

