# MaskFlowNet

## Introduction

<!-- [ALGORITHM] -->

```bibtex
@inproceedings{zhao2020maskflownet,
  title={Maskflownet: Asymmetric feature matching with learnable occlusion mask},
  author={Zhao, Shengyu and Sheng, Yilun and Dong, Yue and Chang, Eric I and Xu, Yan and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6278--6287},
  year={2020}
}
```

## Results and Models

<table>
    <thead>
        <tr>
            <td rowspan=2>Models</td>
            <td rowspan=2>Training datasets</td>
            <td rowspan=2>Flying Chairs</td>
            <td colspan=2>Sintel (training)</td>
            <td colspan=1>KITTI2012 (training)</td>
            <td colspan=1>KITTI2015 (training)</td>
            <td rowspan=2>Log</td>
            <td rowspan=2>Config</td>
            <td rowspan=2>Download</td>
        </tr>
        <tr>
            <th>clean</th>
            <th>final</th>
            <th>EPE</th>
            <th>Fl-all</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th>MaskFlowNet-S</th>
            <th>Flying Chairs</th>
            <th>1.54</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th><a href='https://download.openmmlab.com/mmflow/maskflownet/maskflownets_8x1_slong_flyingchairs_384x448.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/maskflownet/maskflownets_8x1_slong_flyingchairs_384x448.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/maskflownet/maskflownets_8x1_slong_flyingchairs_384x448.pth'>Model</a></th>
        </tr>
        <tr>
            <th>MaskFlowNet-S</th>
            <th>Flying Chairs + Flying Thing3d</th>
            <th>-</th>
            <th>2.30</th>
            <th>3.73</th>
            <th>3.94</th>
            <th>29.70%</th>
            <th><a href='https://download.openmmlab.com/mmflow/maskflownet/maskflownets_8x1_sfine_flyingthings3d_subset_384x768.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/maskflownet/maskflownets_8x1_sfine_flyingthings3d_subset_384x768.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/maskflownet/maskflownets_8x1_sfine_flyingthings3d_subset_384x768.pth'>Model</a></th>
        </tr>
        <tr>
            <th>MaskFlowNet</th>
            <th>Flying Chairs</th>
            <th>1.37</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th><a href='https://download.openmmlab.com/mmflow/maskflownet/maskflownet_8x1_800k_flyingchairs_384x448.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/maskflownet/maskflownet_8x1_800k_flyingchairs_384x448.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/maskflownet/maskflownet_8x1_800k_flyingchairs_384x448.pth'>Model</a></th>
        </tr>
        <tr>
            <th>MaskFlowNet</th>
            <th>Flying Chairs + Flying Thing3d subset</th>
            <th>-</th>
            <th>2.23</th>
            <th>3.70</th>
            <th>3.82</th>
            <th>29.26%</th>
            <th><a href='https://download.openmmlab.com/mmflow/maskflownet/maskflownet_8x1_500k_flyingthings3d_subset_384x768.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/maskflownet/maskflownet_8x1_500k_flyingthings3d_subset_384x768.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/maskflownet/maskflownet_8x1_500k_flyingthings3d_subset_384x768.pth'>Model</a></th>
        </tr>
    </tbody>
</table>
