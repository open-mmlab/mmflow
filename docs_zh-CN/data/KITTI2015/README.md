# 准备 KITTI flow2015 数据集

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

从数据集[官网](http://www.cvlibs.net/datasets/kitti/user_login.php)下载文件压缩包后，解压文件并对照上方目录检查解压后的数据集目录。
