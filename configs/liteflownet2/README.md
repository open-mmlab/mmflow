# LiteFlowNet2

[A Lightweight Optical Flow CNN - Revisiting Data Fidelity and Regularization](https://arxiv.org/abs/1903.07414.pdf)

<!-- [ALGORITHM] -->

## Abstract

Over four decades, the majority addresses the problem of optical flow estimation using variational methods. With the
advance of machine learning, some recent works have attempted to address the problem using convolutional neural network (CNN)
and have showed promising results. FlowNet2, the state-of-the-art CNN, requires over 160M parameters to achieve accurate flow
estimation. Our LiteFlowNet2 outperforms FlowNet2 on Sintel and KITTI benchmarks, while being 25.3 times smaller in the model size
and 3.1 times faster in the running speed. LiteFlowNet2 is built on the foundation laid by conventional methods and resembles the
corresponding roles as data fidelity and regularization in variational methods. We compute optical flow in a spatial-pyramid formulation
as SPyNet but through a novel lightweight cascaded flow inference. It provides high flow estimation accuracy through early
correction with seamless incorporation of descriptor matching. Flow regularization is used to ameliorate the issue of outliers and vague
flow boundaries through feature-driven local convolutions. Our network also owns an effective structure for pyramidal feature extraction
and embraces feature warping rather than image warping as practiced in FlowNet2 and SPyNet. Comparing to LiteFlowNet,
LiteFlowNet2 improves the optical flow accuracy on Sintel Clean by 23.3%, Sintel Final by 12.8%, KITTI 2012 by 19.6%, and KITTI
2015 by 18.8%, while being 2.2 times faster. Our network protocol and trained models are made publicly available on
https://github.com/twhui/LiteFlowNet2.

<div align=center>
<img src="https://user-images.githubusercontent.com/76149310/142731269-eee91f40-1a4d-4c9e-afc6-6d90b0674b62.png" width="70%"/>
</div>

## Results and Models

<table>
    <thead>
        <tr>
            <td rowspan=2>Models</td>
            <td rowspan=2>Training datasets</td>
            <td rowspan=2>FlyingChairs</td>
            <td colspan=2>Sintel (training)</td>
            <td colspan=1>KITTI2012 (training)</td>
            <td colspan=2>KITTI2015 (training)</td>
            <td rowspan=2>Log</td>
            <td rowspan=2>Config</td>
            <td rowspan=2>Download</td>
        </tr>
        <tr>
            <th>clean</th>
            <th>final</th>
            <th>EPE</th>
            <th>Fl-all</th>
            <th>EPE</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th>LiteFlowNet2-pre-M6S6</th>
            <th>Flying Chairs</th>
            <th>4.20</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet2/liteflownet2_pre_M6S6_8x1_flyingchairs_320x448.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet2/liteflownet2_pre_M6S6_8x1_flyingchairs_320x448.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet2/liteflownet2_pre_M6S6_8x1_flyingchairs_320x448.pth'>Model</a></th>
        </tr>
        <tr>
            <th>LiteFlowNet2-pre-M6S6R6</th>
            <th>Flying Chairs</th>
            <th>3.94</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet2/liteflownet2_pre_M6S6R6_8x1_flyingchairs_320x448.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet2/liteflownet2_pre_M6S6R6_8x1_flyingchairs_320x448.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet2/liteflownet2_pre_M6S6R6_8x1_flyingchairs_320x448.pth'>Model</a></th>
        </tr>
        <tr>
            <th>LiteFlowNet2-pre-M5S5R5</th>
            <th>Flying Chairs</th>
            <th>2.85</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet2/liteflownet2_pre_M5S5R5_8x1_flyingchairs_320x448.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet2/liteflownet2_pre_M5S5R5_8x1_flyingchairs_320x448.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet2/liteflownet2_pre_M5S5R5_8x1_flyingchairs_320x448.pth'>Model</a></th>
        </tr>
        <tr>
            <th>LiteFlowNet2-pre-M4S4R4</th>
            <th>Flying Chairs</th>
            <th>2.07</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet2/liteflownet2_pre_M4S4R4_8x1_flyingchairs_320x448.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet2/liteflownet2_pre_M4S4R4_8x1_flyingchairs_320x448.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet2/liteflownet2_pre_M4S4R4_8x1_flyingchairs_320x448.pth'>Model</a></th>
        </tr>
        <tr>
            <th>LiteFlowNet2-pre (LiteFlowNet2-pre-M3S3R3)</th>
            <th>Flying Chairs</th>
            <th>1.57</th>
            <th>2.78</th>
            <th>4.24</th>
            <th>5.79</th>
            <th>39.42%</th>
            <th>14.34</th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet2/liteflownet2_pre_M3S3R3_8x1_flyingchairs_320x448.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet2/liteflownet2_pre_M3S3R3_8x1_flyingchairs_320x448.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet2/liteflownet2_pre_M3S3R3_8x1_flyingchairs_320x448.pth'>Model</a></th>
        </tr>
        <tr>
            <th>LiteFlowNet2</th>
            <th>Flying Chairs + Flying Thing3d subset</th>
            <th>-</th>
            <th>2.35</th>
            <th>3.86</th>
            <th>4.84</th>
            <th>32.87%</th>
            <th>12.07</th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet2/liteflownet2_8x1_500k_flyingthing3d_subset_384x768.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet2/liteflownet2_8x1_500k_flyingthing3d_subset_384x768.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet2/liteflownet2_8x1_500k_flyingthing3d_subset_384x768.pth'>Model</a></th>
        </tr>
        <tr>
            <th>LiteFlowNet2</th>
            <th>Flying Chairs + Flying Thing3d subset + Sintel + KITTI</th>
            <th>-</th>
            <th>1.32</th>
            <th>1.65</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet2/liteflownet2_ft_4x1_600k_sintel_kitti_320x768.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet2/liteflownet2_ft_4x1_600k_sintel_kitti_320x768.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet2/liteflownet2_ft_4x1_600k_sintel_kitti_320x768.pth'>Model</a></th>
        </tr>
        <tr>
            <th>LiteFlowNet2</th>
            <th>Flying Chairs + Flying Thing3d subset + KITTI</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th>0.89</th>
            <th>4.31%</th>
            <th>1.24</th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet2/liteflownet2_ft_4x1_500k_kitti_320x896.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet2/liteflownet2_ft_4x1_500k_kitti_320x896.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet2/liteflownet2_ft_4x1_500k_kitti_320x896.pth'>Model</a></th>
        </tr>
    </tbody>
</table>

## Citation

```bibtex
@article{hui2020lightweight,
  title={A lightweight optical flow CNN—Revisiting data fidelity and regularization},
  author={Hui, Tak-Wai and Tang, Xiaoou and Loy, Chen Change},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={43},
  number={8},
  pages={2555--2569},
  year={2020},
  publisher={IEEE}
}
```
