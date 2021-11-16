# Prepare KITTI flow2012 dataset

<!-- [DATASET] -->

```bibtex
@INPROCEEDINGS{Geiger2012CVPR,
  author = {Andreas Geiger and Philip Lenz and Raquel Urtasun},
  title = {Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2012}
}
```

```text
kitti2012
|   |   ├── training
|   |   |   ├── flow_occ
|   |   |   |   ├── xxxxxx_xx.png
|   |   |   ├── flow_noc
|   |   |   |   ├── xxxxxx_xx.png
|   |   |   ├── colored_0
|   |   |   |   ├── xxxxxx_xx.png
```

You can download datasets on this [webpage](http://www.cvlibs.net/datasets/kitti/user_login.php). Then, you need to unzip and move corresponding datasets to follow the folder structure shown above. The datasets have been well-prepared by the original authors.
