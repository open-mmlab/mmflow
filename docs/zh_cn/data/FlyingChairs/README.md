# 准备 FlyingChairs 数据集

<!-- [DATASET] -->

```bibtex
@InProceedings{DFIB15,
  author    = "A. Dosovitskiy and P. Fischer and E. Ilg and P. H{\"a}usser and C. Haz{\i}rba{\c{s}} and V. Golkov and P. v.d. Smagt and D. Cremers and T. Brox",
  title     = "FlowNet: Learning Optical Flow with Convolutional Networks",
  booktitle = "IEEE International Conference on Computer Vision (ICCV)",
  month     = " ",
  year      = "2015",
  url       = "http://lmb.informatik.uni-freiburg.de/Publications/2015/DFIB15"
}
```

```text

├── FlyingChairs_release
│   ├── FlyingChairs_train_val.txt
|   ├── data
|   |    ├── xxxxx_flow.flo
|   |    ├── xxxxx_img1.ppm
|   |    ├── xxxxx_img2.ppm
```

从数据集[官网](https://lmb.informatik.uni-freiburg.de/data/FlyingChairs/FlyingChairs.zip)下载文件压缩包后，解压文件并对照上方目录检查解压后的数据集目录。之后，下载 [训练-验证拆分文件](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs/FlyingChairs_train_val.txt),并放置与 `FlyingChairs_release` 文件夹下。
