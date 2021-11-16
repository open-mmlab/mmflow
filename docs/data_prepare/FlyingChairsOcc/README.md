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

Here is the script to prepare FlyingChairsOcc dataset.

```bash
wget https://download.visinf.tu-darmstadt.de/data/flyingchairs_occ/FlyingChairsOcc.tar.gz
tar -xvf FlyingChairsOcc.tar.gz
```
