import torch, numpy as np, open3d as o3d, pandas as pd

"""
This file is to go through a semantic image file and just color
one single label. It is ONLY to color one single label
March 9th, Jacob Claessens
"""


observations = torch.load("train_observations_lst.pt")
step_index = 3#8#3#8 #3 #8 #15 #8 #3
semantic_image = observations[step_index]['semantic'].squeeze()
 
"""
In [81]: uni, counts = np.unique(semantic_image, return_counts=True)

In [82]: uni
Out[82]: 
array([  0, 642, 644, 646, 675, 676, 696, 709, 713, 714, 715, 716, 717,
       718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 730, 731,
       732, 733, 734, 737, 741, 743, 746, 747, 748, 749, 750, 751, 752,
       753, 754, 755, 756, 757, 758, 759, 764, 765], dtype=int32)

In [83]: counts
Out[83]: 
array([ 1558,  1006,    91,   138,  2057,   383,  5919,   377,  7364,
        2353,  2937, 10394, 11032,   639,   261,    48,   287,   663,
        1564,  1733,  2642,   602,   630,   667,   438,   100,   447,
          34,   678,     7,     5, 35927, 25273,  7208,   424,   514,
        5547,   745, 16324,  3704, 27623, 76450,   604,   200, 49453,
          33,    82,    35])

"""
img_rgb_lst = []
for x in semantic_image:
    img_rgb_sub_lst = []
    for y in x:
        if y == 758:
            print("ENTERED")
            img_rgb_sub_lst.append([122, 122, 122])
        else:
            img_rgb_sub_lst.append([255, 255, 255])
    img_rgb_lst.append(img_rgb_sub_lst)

img_rgb_arr = np.array(img_rgb_lst).astype(np.uint8)  
visualizer = o3d.visualization.Visualizer()
visualizer.create_window()
image = o3d.geometry.Image(img_rgb_arr)
visualizer.add_geometry(image)

# Set camera view
visualizer.get_render_option().background_color = np.asarray([0, 0, 0])
visualizer.get_view_control().set_zoom(0.5)
visualizer.run()

    

