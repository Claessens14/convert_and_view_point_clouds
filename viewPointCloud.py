import numpy as np; import pptk; import laspy; import torch; import os; import pathlib; import argparse

# IMPORTANT - pptk requires python 3.6 to run on MAC
# --> conda activate python3_6_mac
# ***clf - classified***
# **1.7 - version of release used**
# **use numpy.c_**
# rsync -avzrth -e ssh claess14@niagara.scinet.utoronto.ca:/scratch/l/leil/claess14/sparse_control/imgs/REIN_percept_walk_ptcld/xp160/REIN_percept_walk_ptcld_5239860122177.pt      ./multi_ngbh_data_new
# Files failing:   
# 5145370841675.las  5239860122177.las     
# 5145370841675.las  5239860122177.las    
# 5145370841675.las  5239860122177.las   

def get_aethon_rgb(class_ids):
    if not isinstance(class_ids, np.ndarray):
        # assume it's a torch tensor
        class_ids = class_ids.numpy()
    #ground
    rgb = (class_ids == 0)[:, np.newaxis].astype(np.float32)*np.array([[153, 153, 102]])/255
    # veg
    rgb +=(class_ids == 1)[:, np.newaxis].astype(np.float32)*np.array([[0,153, 0]])/255
    # poles
    rgb +=(class_ids == 2)[:, np.newaxis].astype(np.float32)*np.array([[10, 20, 250]])/255
    # powerlines
    rgb +=(class_ids == 3)[:, np.newaxis].astype(np.float32)*np.array([[250, 10, 10]])/255
    # other wires
    rgb +=(class_ids == 4)[:, np.newaxis].astype(np.float32)*np.array([[204, 204, 10]])/255
    # building
    rgb +=(class_ids == 5)[:, np.newaxis].astype(np.float32)*np.array([[204, 102, 0]])/255
    return rgb 


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", default='FALSE')
    parser.add_argument("--xp_id", default="xp166")
    parser.add_argument("--point_size", default=0.02)
    args = parser.parse_args()
    print(args) 

    dir_path = f"REIN_percept_walk_ptcld_local/{args.xp_id}"
    print(args.xp_id)
    print(f"dir_path to local files:  {dir_path}")
    
    if args.download == 'TRUE': 
        print("\nThis will be needed:   dfpqlDWOmxe8273443")
        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
        os.system(f"rsync -avzrth -e ssh claess14@niagara.scinet.utoronto.ca:/scratch/l/leil/claess14/sparse_control/imgs/REIN_percept_walk_ptcld/{args.xp_id}/*  ./{dir_path}")
    
    f = []
    for (dirpath, dirnames, filenames) in os.walk(dir_path):
        f.append(filenames)
        break

    # view = pptk.viewer(np.c_[fp.x, fp.y, fp.z])
    # multi_ngbh_data/multi_ngbh_data_torch_save.pt

    for file_name in f[0]:
        ngbh = torch.load(dir_path + "/" + file_name)
        import ipdb; ipdb.set_trace()
        #view = pptk.viewer(ngbh['ngbh_cat_short'], title=f"{args.xp_id}/{file_name}")
        view = pptk.viewer(ngbh['pos'], title=f"{args.xp_id}/{file_name}")
        # data = get_aethon_rgb(fp.raw_classification)
        data = get_aethon_rgb(ngbh['labels'])
        #print(f"ngbh['ngbh_cat_short']:  {ngbh['ngbh_cat_short'].shape}")
        #print(f"ngbh['labels']:  {ngbh['labels'].shape}")
        view.attributes(data)
        view.set(point_size=args.point_size)
