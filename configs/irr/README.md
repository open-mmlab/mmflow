# IRR

## Introduction

<!-- [ALGORITHM] -->

```bibtex
@inproceedings{hur2019iterative,
  title={Iterative residual refinement for joint optical flow and occlusion estimation},
  author={Hur, Junhwa and Roth, Stefan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5754--5763},
  year={2019}
}
```

## Results and Models

<table>
    <thead>
        <tr>
            <td rowspan=2>Models</td>
            <td rowspan=2>Training datasets</td>
            <td rowspan=2>FlyingChairsOcc</td>
            <td colspan=2>Sintel (training)</td>
            <td colspan=2>KITTI2015 (training)</td>
            <td rowspan=2>Log</td>
            <td rowspan=2>Config</td>
            <td rowspan=2>Download</td>
        </tr>
        <tr>
            <th>clean</th>
            <th>final</th>
            <th>Fl-all</th>
            <th>EPE</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th>IRR-PWC</th>
            <th>FlyingChairsOcc</th>
            <th>1.44</th>
            <th>2.38</th>
            <th>3.86</th>
            <th>-</th>
            <th>-</th>
            <th><a href='https://download.openmmlab.com/mmflow/irr/irrpwc_8x1_sshort_flyingchairsocc_384x448.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/irr/irrpwc_8x1_sshort_flyingchairsocc_384x448.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/irr/irrpwc_8x1_sshort_flyingchairsocc_384x448.pth'>Model</a></th>
        </tr>
        <tr>
            <th>IRR-PWC</th>
            <th>FlyingChairsOcc + FlyingThing3d subset</th>
            <th>-</th>
            <th>1.79</th>
            <th>3.38</th>
            <th>25.06%</th>
            <th>8.32</th>
            <th><a href='https://download.openmmlab.com/mmflow/irr/irrpwc_8x1_sfine_half_flyingthings3d_subset_384x768.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/irr/irrpwc_8x1_sfine_half_flyingthings3d_subset_384x768.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/irr/irrpwc_8x1_sfine_half_flyingthings3d_subset_384x768.pth'>Model</a></th>
        </tr>
        <tr>
            <th>IRR-PWC-ft</th>
            <th>FlyingChairsOcc + FlyingThing3d subset + Sintel</th>
            <th>-</th>
            <th>1.51</th>
            <th>2.18</th>
            <th>-</th>
            <th>-</th>
            <th><a href='https://download.openmmlab.com/mmflow/irr/irrpwc_ft_4x1_300k_sintel_384x768.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/irr/irrpwc_ft_4x1_300k_sintel_384x768.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/irr/irrpwc_ft_4x1_300k_sintel_384x768.pth'>Model</a></th>
        </tr>
        <tr>
            <th>IRR-PWC-ft-final</th>
            <th>FlyingChairsOcc + FlyingThing3d subset + Sintel final</th>
            <th>-</th>
            <th>1.71</th>
            <th>1.94</th>
            <th>-</th>
            <th>-</th>
            <th><a href='https://download.openmmlab.com/mmflow/irr/irrpwc_ft_4x1_300k_sintel_final_384x768.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/irr/irrpwc_ft_4x1_300k_sintel_final_384x768.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/irr/irrpwc_ft_4x1_300k_sintel_final_384x768.pth'>Model</a></th>
        </tr>
        <tr>
            <th>IRR-PWC-ft</th>
            <th>FlyingChairsOcc + FlyingThing3d subset + KITTI</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th>8.51%</th>
            <th>2.19</th>
            <th><a href='https://download.openmmlab.com/mmflow/irr/irrpwc_ft_4x1_300k_kitti_320x896.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/irr/irrpwc_ft_4x1_300k_kitti_320x896.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/irr/irrpwc_ft_4x1_300k_kitti_320x896.pth'>Model</a></th>
        </tr>
    </tbody>
</table>
