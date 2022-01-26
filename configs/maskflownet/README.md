# MaskFlowNet

[MaskFlownet: Asymmetric Feature Matching with Learnable Occlusion Mask](https://arxiv.org/pdf/2003.10955.pdf)

<!-- [ALGORITHM] -->

## Abstract

Feature warping is a core technique in optical flow estimation;
however, the ambiguity caused by occluded areas during warping is a major
problem that remains unsolved. In this paper, we propose
an asymmetric occlusionaware feature matching module,
which can learn a rough occlusion mask that filters useless (occluded) areas
immediately after feature warping without any explicit supervision.
The proposed module can be easily integrated into
end-to-end network architectures and enjoys performance
gains while introducing negligible computational cost. The
learned occlusion mask can be further fed into a subsequent
network cascade with dual feature pyramids with which we
achieve state-of-the-art performance. At the time of submission,
our method, called MaskFlownet, surpasses all published optical flow
methods on the MPI Sintel, KITTI 2012 and 2015 benchmarks.
Code is available at https://github.com/microsoft/MaskFlownet.

<div align=center>
<img src="https://user-images.githubusercontent.com/76149310/142731471-ed5fc41b-59f9-4e00-b27b-d0456b2a09a2.png" width="70%"/>
</div>

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

## Citation

```bibtex
@inproceedings{zhao2020maskflownet,
  title={Maskflownet: Asymmetric feature matching with learnable occlusion mask},
  author={Zhao, Shengyu and Sheng, Yilun and Dong, Yue and Chang, Eric I and Xu, Yan and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6278--6287},
  year={2020}
}
```
