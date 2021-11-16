# LiteFlowNet2

## Introduction

<!-- [ALGORITHM] -->

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
