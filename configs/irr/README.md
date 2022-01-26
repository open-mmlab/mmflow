# IRR

[Iterative Residual Refinement for Joint Optical Flow and Occlusion Estimation](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hur_Iterative_Residual_Refinement_for_Joint_Optical_Flow_and_Occlusion_Estimation_CVPR_2019_paper.pdf)

<!-- [ALGORITHM] -->

## Abstract

Deep learning approaches to optical flow estimation
have seen rapid progress over the recent years. One common trait of
many networks is that they refine an initial flow estimate either
through multiple stages or across the levels of a coarse-to-fine representation.
While leading to more accurate results, the downside of this is an increased
number of parameters. Taking inspiration from both classical
energy minimization approaches as well as residual
networks, we propose an iterative residual refinement (IRR)
scheme based on weight sharing that can be combined with
several backbone networks. It reduces the number of parameters,
improves the accuracy, or even achieves both. Moreover,
we show that integrating occlusion prediction and bi-directional
flow estimation into our IRR scheme can
further boost the accuracy. Our full network achieves state-
of-the-art results for both optical flow
and occlusion estimation across several standard datasets.

<div align=center>
<img src="https://user-images.githubusercontent.com/76149310/142731424-9cda1d89-e222-4bcf-b1b8-b18b31f7643b.png" width="70%"/>
</div>

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

## Citation

```bibtex
@inproceedings{hur2019iterative,
  title={Iterative residual refinement for joint optical flow and occlusion estimation},
  author={Hur, Junhwa and Roth, Stefan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5754--5763},
  year={2019}
}
```
