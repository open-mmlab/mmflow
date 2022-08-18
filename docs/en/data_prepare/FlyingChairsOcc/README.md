# Prepare FlyingChairsOcc dataset

<!-- [DATASET] -->

```bibtex
@inproceedings{Hur:2019:IRR,
  Author = {Junhwa Hur and Stefan Roth},
  Booktitle = {CVPR},
  Title = {Iterative Residual Refinement for Joint Optical Flow and Occlusion Estimation},
  Year = {2019}
}
```

## Download and Unpack dataset

Please download the datasets from the official websites.

```bash
wget https://download.visinf.tu-darmstadt.de/data/flyingchairs_occ/FlyingChairsOcc.tar.gz
tar -xvf FlyingChairsOcc.tar.gz
```

If your dataset folder structure is different from the following, you may need to change the corresponding paths.

```text

├── FlyingChairsOcc
|   |   ├── data
|   |   |    ├── xxxxx_flow.flo
|   |   |    ├── xxxxx_flow_b.flo
|   |   |    ├── xxxxx_img1.ppm
|   |   |    ├── xxxxx_img2.ppm
|   |   |    ├── xxxxx_occ1.png
|   |   |    ├── xxxxx_occ2.png
```

## Generate annotation file

We provide a convenient script to generate annotation file, which list all of data samples in the dataset.
You can use the following command to generate annotation file.
