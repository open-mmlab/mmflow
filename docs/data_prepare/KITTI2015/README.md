# Prepare KITTI flow2015 dataset

<!-- [DATASET] -->

```bibtex
@ARTICLE{Menze2018JPRS,
  author = {Moritz Menze and Christian Heipke and Andreas Geiger},
  title = {Object Scene Flow},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing (JPRS)},
  year = {2018}
}

@INPROCEEDINGS{Menze2015ISA,
  author = {Moritz Menze and Christian Heipke and Andreas Geiger},
  title = {Joint 3D Estimation of Vehicles and Scene Flow},
  booktitle = {ISPRS Workshop on Image Sequence Analysis (ISA)},
  year = {2015}
}
```

```text
kitti2015
|   |   ├── training
|   |   |   ├── flow_occ
|   |   |   |   ├── xxxxxx_xx.png
|   |   |   ├── flow_noc
|   |   |   |   ├── xxxxxx_xx.png
|   |   |   ├── image_2
|   |   |   |   ├── xxxxxx_xx.png
```

You can download datasets on this [webpage](http://www.cvlibs.net/datasets/kitti/user_login.php). Then, you need to unzip and move corresponding datasets to follow the folder structure shown above. The datasets have been well-prepared by the original authors.
