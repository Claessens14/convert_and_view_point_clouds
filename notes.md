# Image to Point Cloud

### Dev

- [ ] convert the image/depth information to points
    - [x] crude way
`pip install open3d `
    - [ ] standardized algorithm that does SLAM (multiImage)
```
devendrra
depth_utils
https://github.com/devendrachaplot/Object-Goal-Navigation/blob/5d76902fe9be821926a1de32557ca9a8dc21d0f5/envs/utils/depth_utils.py

colours semantics constants
https://github.com/devendrachaplot/Object-Goal-Navigation/blob/5d76902fe9be821926a1de32557ca9a8dc21d0f5/constants.py

geo-centric point cloud creation
https://github.com/devendrachaplot/Object-Goal-Navigation/blob/master/model.py
and usuage
https://github.com/devendrachaplot/Object-Goal-Navigation/blob/master/model.py#L188

aux task semantic labels
sofa green
chair yellowish orange
wall orange
window bright yellow
https://github.com/niessner/Matterport/blob/master/metadata/mpcat40.tsv

```





```
rsync -avzrth -e ssh claess14@niagara.scinet.utoronto.ca:/scratch/l/leil/claess14/habitat-challenge-v3/habitat-lab/output/train_observations_lst.pt .
```

```
use python not python3 for torch to work
python3_6_mac
Desktop/machine-learning/rsync
```




