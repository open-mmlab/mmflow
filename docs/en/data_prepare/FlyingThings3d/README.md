# Prepare FlyingThing3d dataset

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

You can download datasets via \[BitTorrent\] (https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/raw_data/flyingthings3d__frames_cleanpass.tar.torrent). Then, you need to unzip and move corresponding datasets to follow the folder structure shown above. The datasets have been well-prepared by the original authors.
