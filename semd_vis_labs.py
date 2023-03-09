import torch, numpy as np, open3d as o3d, pandas as pd

"""
This file is to simply open a semD image, and show a FEW categories. This includes
wall, floow, ceiling, window, door
Jacob Claessens March 9th
"""

observations = torch.load("train_observations_lst.pt")
step_index = 3#8#3#8 #3 #8 #15 #8 #3
semantic_image = observations[step_index]['semantic'].squeeze()

import colorsys
def make_rgb_palette(n=40):
    HSV_tuples = [(x*1.0/n, 0.8, 0.8) for x in range(n)]
    RGB_map = np.array(list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)))
    return RGB_map
colors = make_rgb_palette(45)
# label is a target label, keep it, if now, change it to 1000
sem_matx_lst = []
tsv_name = 'matterport_category_mappings'; df = pd.read_csv(tsv_name+'.tsv', sep='  +')
for x in semantic_image:
    sub_sem_lst = []
    for y in x:
        sub_sem_lst.append(y if df['category'][df['index'] == y+1].values[0] in ['wall', 'door', 'floor', 'ceiling'] else 1000)
    sem_matx_lst.append(sub_sem_lst)
semantic_image = np.array(sem_matx_lst)
semantic_colors = colors[semantic_image % 45] * 255
semantic_colors = semantic_colors.astype(np.uint8)
visualizer = o3d.visualization.Visualizer()
visualizer.create_window()
image = o3d.geometry.Image(semantic_colors)
visualizer.add_geometry(image)

# Set camera view
visualizer.get_render_option().background_color = np.asarray([0, 0, 0])
visualizer.get_view_control().set_zoom(0.5)
visualizer.run()





