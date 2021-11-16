# 准备 FlyingThing3d_subset 数据集

<!-- [DATASET] -->

```bibtex
@InProceedings{MIFDB16,
  author    = "N. Mayer and E. Ilg and P. H{\"a}usser and P. Fischer and D. Cremers and A. Dosovitskiy and T. Brox",
  title     = "A Large Dataset to Train Convolutional Networks for Disparity, Optical Flow, and Scene Flow Estimation",
  booktitle = "IEEE International Conference on Computer Vision and Pattern Recognition (CVPR)",
  year      = "2016",
  note      = "arXiv:1512.02134",
  url       = "http://lmb.informatik.uni-freiburg.de/Publications/2016/MIFDB16"
}
```

```text
├── FlyingThings3D_subset
|   |   ├── train
|   |   |   ├── flow
|   |   |   |   ├── left
|   |   |   |   |    ├── into_future
|   |   |   |   |    |      ├── xxxxxxx.flo
|   |   |   |   |    ├── into_past
|   |   |   |   |    |      ├── xxxxxxx.flo
|   |   |   |   ├── right
|   |   |   |   |    ├── into_future
|   |   |   |   |    |      ├── xxxxxxx.flo
|   |   |   |   |    ├── into_past
|   |   |   |   |    |      ├── xxxxxxx.flo
|   |   |   ├── flow_occlusions
|   |   |   |   ├── left
|   |   |   |   |    ├── into_future
|   |   |   |   |    |      ├── xxxxxxx.flo
|   |   |   |   |    ├── into_past
|   |   |   |   |    |      ├── xxxxxxx.flo
|   |   |   |   ├── right
|   |   |   |   |    ├── into_future
|   |   |   |   |    |      ├── xxxxxxx.flo
|   |   |   |   |    ├── into_past
|   |   |   |   |    |      ├── xxxxxxx.flo
|   |   |   ├── image_clean
|   |   |   |   ├── left
|   |   |   |   |    ├── xxxxxxx.png
|   |   |   |   ├── right
|   |   |   |   |    ├── xxxxxxx.png
|   |   ├── val
|   |   |   ├── flow
|   |   |   |   ├── left
|   |   |   |   |    ├── into_future
|   |   |   |   |    |      ├── xxxxxxx.flo
|   |   |   |   |    ├── into_past
|   |   |   |   |    |      ├── xxxxxxx.flo
|   |   |   |   ├── right
|   |   |   |   |    ├── into_future
|   |   |   |   |    |      ├── xxxxxxx.flo
|   |   |   |   |    ├── into_past
|   |   |   |   |    |      ├── xxxxxxx.flo
|   |   |   ├── flow_occlusions
|   |   |   |   ├── left
|   |   |   |   |    ├── into_future
|   |   |   |   |    |      ├── xxxxxxx.flo
|   |   |   |   |    ├── into_past
|   |   |   |   |    |      ├── xxxxxxx.flo
|   |   |   |   ├── right
|   |   |   |   |    ├── into_future
|   |   |   |   |    |      ├── xxxxxxx.flo
|   |   |   |   |    ├── into_past
|   |   |   |   |    |      ├── xxxxxxx.flo
|   |   |   ├── image_clean
|   |   |   |   ├── left
|   |   |   |   |    ├── xxxxxxx.png
|   |   |   |   ├── right
|   |   |   |   |    ├── xxxxxxx.png
```

下载 [数据集](https://lmb.informatik.uni-freiburg.de/data/FlyingThings3D_subset/FlyingThings3D_subset_image_clean.tar.bz2.torrent) 的 BitTorrent。通过 BitTorrent 下载并解压到 `FlyingThings3D_subset`
