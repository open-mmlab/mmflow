# Prepare FlyingChairs dataset

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

Here is the script to prepare FlyingChairs dataset.

```bash
wget https://lmb.informatik.uni-freiburg.de/data/FlyingChairs/FlyingChairs.zip
unzip FlyingChairs.zip
cd FlyingChairs_release
wget https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs/FlyingChairs_train_val.txt
```
