# 准备 FlyingThing3d 数据集

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
├── flyingthings3d
|   |   ├── frames_cleanpass
|   |   |   ├── TEST
|   |   |   |   ├── x
|   |   |   |   |   ├── xxxx
|   |   |   |   |   |    ├── left
|   |   |   |   |   |    |   ├── xxxx.png
|   |   |   |   |   |    ├── right
|   |   |   |   |   |    |   ├── xxxx.png
|   |   |   ├── TRAIN
|   |   |   |   ├── x
|   |   |   |   |   ├── xxxx
|   |   |   |   |   |    ├── left
|   |   |   |   |   |    |   ├── xxxx.png
|   |   |   |   |   |    ├── right
|   |   |   |   |   |    |   ├── xxxx.png
|   |   ├── frames_finalpass
|   |   |   ├── TEST
|   |   |   |   ├── x
|   |   |   |   |   ├── xxxx
|   |   |   |   |   |    ├── left
|   |   |   |   |   |    |   ├── xxxx.png
|   |   |   |   |   |    ├── right
|   |   |   |   |   |    |   ├── xxxx.png
|   |   |   ├── TRAIN
|   |   |   |   ├── x
|   |   |   |   |   ├── xxxx
|   |   |   |   |   |    ├── left
|   |   |   |   |   |    |   ├── xxxx.png
|   |   |   |   |   |    ├── right
|   |   |   |   |   |    |   ├── xxxx.png
|   |   ├── optical_flow
|   |   |   ├── TEST
|   |   |   |   ├── x
|   |   |   |   |   ├── xxxx
|   |   |   |   |   |    ├── into_future
|   |   |   |   |   |    |       ├── left
|   |   |   |   |   |    |       |     ├── OpticalFlowIntoFuture_xxxx_L.pfm
|   |   |   |   |   |    |       ├── right
|   |   |   |   |   |    |       |     ├── OpticalFlowIntoFuture_xxxx_R.pfm
|   |   |   |   |   |    ├── into_past
|   |   |   |   |   |    |       ├── left
|   |   |   |   |   |    |       |     ├── OpticalFlowIntoPast_xxxx_L.pfm
|   |   |   |   |   |    |       ├── right
|   |   |   |   |   |    |       |     ├── OpticalFlowIntoPast_xxxx_R.pfm
|   |   |   ├── TRAIN
|   |   |   |   ├── x
|   |   |   |   |   ├── xxxx
|   |   |   |   |   |    ├── into_future
|   |   |   |   |   |    |       ├── left
|   |   |   |   |   |    |       |     ├── OpticalFlowIntoFuture_xxxx_L.pfm
|   |   |   |   |   |    |       ├── right
|   |   |   |   |   |    |       |     ├── OpticalFlowIntoFuture_xxxx_R.pfm
|   |   |   |   |   |    ├── into_past
|   |   |   |   |   |    |       ├── left
|   |   |   |   |   |    |       |     ├── OpticalFlowIntoPast_xxxx_L.pfm
|   |   |   |   |   |    |       ├── right
|   |   |   |   |   |    |       |     ├── OpticalFlowIntoPast_xxxx_R.pfm
```

下载 [clean pass](https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/raw_data/flyingthings3d__frames_cleanpass.tar.torrent) 和 [final pass](https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/raw_data/flyingthings3d__frames_finalpass.tar.torrent) 数据，和 [flow map](https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/derived_data/flyingthings3d__optical_flow.tar.bz2.torrent)的 BitTorrent。通过 BitTorrent 下载并解压到 `frames_cleanpass` `frames_finalpass` 和 `optical_flow`　目录下
