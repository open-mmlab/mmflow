# Prepare ChairsSDHom dataset

<!-- [DATASET] -->

```bibtex
@InProceedings{IMKDB17,
  author    = "E. Ilg and N. Mayer and T. Saikia and M. Keuper and A. Dosovitskiy and T. Brox",
  title     = "FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks",
  booktitle = "IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
  month     = "Jul",
  year      = "2017",
  url       = "http://lmb.informatik.uni-freiburg.de//Publications/2017/IMKDB17"
}
```

```text
ChairsSDHom
|   |   ├── data
|   |   |    ├── train
|   |   |    |    |── flow
|   |   |    |    |      |── xxxxx.pfm
|   |   |    |    |── t0
|   |   |    |    |      |── xxxxx.png
|   |   |    |    |── t1
|   |   |    |    |      |── xxxxx.png
|   |   |    ├── test
|   |   |    |    |── flow
|   |   |    |    |      |── xxxxx.pfm
|   |   |    |    |── t0
|   |   |    |    |      |── xxxxx.png
|   |   |    |    |── t1
|   |   |    |    |      |── xxxxx.png
```

Here is the script to prepare ChairsSDHom dataset.

```bash
wget https://lmb.informatik.uni-freiburg.de/data/FlowNet2/ChairsSDHom/ChairsSDHom.tar.gz
tar -xvf ChairsSDHom.tar.gz
```
