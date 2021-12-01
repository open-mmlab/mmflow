# 准备 Sintel 数据集

<!-- [DATASET] -->

```bibtex
@inproceedings{Butler:ECCV:2012,
title = {A naturalistic open source movie for optical flow evaluation},
author = {Butler, D. J. and Wulff, J. and Stanley, G. B. and Black, M. J.},
booktitle = {European Conf. on Computer Vision (ECCV)},
editor = {{A. Fitzgibbon et al. (Eds.)}},
publisher = {Springer-Verlag},
series = {Part IV, LNCS 7577},
month = oct,
pages = {611--625},
year = {2012}
}

@inproceedings{Wulff:ECCVws:2012,
 title = {Lessons and insights from creating a synthetic optical flow benchmark},
 author = {Wulff, J. and Butler, D. J. and Stanley, G. B. and Black, M. J.},
 booktitle = {ECCV Workshop on Unsolved Problems in Optical Flow and Stereo Estimation},
 editor = {{A. Fusiello et al. (Eds.)}},
 publisher = {Springer-Verlag},
 series = {Part II, LNCS 7584},
 month = oct,
 pages = {168--177},
 year = {2012}
}
```

```text
 Sintel
|   |   ├── training
|   |   |   ├── clean
|   |   |   |   ├── xxxx_x
|   |   |   |   |    ├── frame_xxxx.png
|   |   |   ├── final
|   |   |   |   ├── xxxx_x
|   |   |   |   |    ├── frame_xxxx.png
|   |   |   ├── flow
|   |   |   |   |    ├── frame_xxxx.flo
|   |   |   ├── invalid
|   |   |   |   ├── xxxx_x
|   |   |   |   |    ├── frame_xxxx.png
```

从数据集提供的 [European mirror](http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip), 和 [US mirror](http://sintel.cs.washington.edu/MPI-Sintel-complete.zip) 下载文件压缩包后，解压文件并对照上方目录检查解压后的数据集目录。
