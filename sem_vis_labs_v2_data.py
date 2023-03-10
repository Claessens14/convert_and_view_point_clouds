import torch, numpy as np, open3d as o3d, pandas as pd

"""
This file is a copy of semD_vis_labs.py, this is to test out the new label table, that is aligned with 
the env we are loadeding.
 - [ ] may not need plus one for y
 - [ ] need to check indexing
Jacob Claessens March 9th
"""

observations = torch.load("train_observations_lst_v2.pt")
step_index = 3#8#3#8 #3 #8 #15 #8 #3
semantic_image = observations[step_index]['semantic'].squeeze()

import colorsys
def make_rgb_palette(n=40):
    HSV_tuples = [(x*1.0/n, 0.8, 0.8) for x in range(n)]
    RGB_map = np.array(list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)))
    return RGB_map
colors = make_rgb_palette(45)
# label is a target label, keep it, if now, change it to 1000
sem_matx_lst = []; label_name = 'category'
df = pd.read_csv('gjhYih4upQ9.semantic.txt', sep=',')

for x in semantic_image:
    sub_sem_lst = []
    for y in x:
        sub_sem_lst.append(y+1 if df[label_name][df['index'] == y+1].str.contains('wall|door|floor|ceiling').values else 20)
    sem_matx_lst.append(sub_sem_lst)

def colored_background(red, g, b, text):
    # https://stackoverflow.com/questions/70519979/printing-with-rgb-background
    # return a string that shows background rgb colours in bash terminals. may need tmux to show
    return f"\033[48;2;{red};{g};{b}m{text}\033[0m"

sem_unique_idx_lst = [el for el in set([el  for row in sem_matx_lst for el in row])]
sem_unique_label_lst = df.loc[df['index'].isin(sem_unique_idx_lst), label_name].values.tolist() #df[label_name][df['index']==sem_unique_idx_lst].values.tolist()
sem_unique_col_lst = colors[np.array(sem_unique_idx_lst) % 45] * 255
for i in range(len(sem_unique_col_lst)):
    red, g, b = sem_unique_col_lst[i][0], sem_unique_col_lst[i][1], sem_unique_col_lst[i][2]
    red, g, b = int(red), int(g), int(b)
    if sem_unique_idx_lst[i] == 20:
        text = "    OTHER"
    else: 
        text = str(sem_unique_idx_lst[i])+" "+sem_unique_label_lst[i] 
    print(colored_background(red, g, b, text))
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





