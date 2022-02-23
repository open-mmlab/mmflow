# LiteFlowNet

[LiteFlowNet: A Lightweight Convolutional Neural Network for Optical Flow Estimation](https://arxiv.org/pdf/1805.07036.pdf)

<!-- [ALGORITHM] -->

## Abstract

FlowNet2, the state-of-the-art convolutional neural
network (CNN) for optical flow estimation, requires over
160M parameters to achieve accurate flow estimation. In
this paper we present an alternative network that outperforms
FlowNet2 on the challenging Sintel final pass and
KITTI benchmarks, while being 30 times smaller in the
model size and 1.36 times faster in the running speed. This
is made possible by drilling down to architectural details
that might have been missed in the current frameworks: (1)
We present a more effective flow inference approach at each
pyramid level through a lightweight cascaded network. It
not only improves flow estimation accuracy through early
correction, but also permits seamless incorporation of descriptor matching
in our network. (2) We present a novel flow regularization layer
to ameliorate the issue of outliers and vague flow boundaries
by using a feature-driven local convolution. (3) Our network owns
an effective structure for pyramidal feature extraction and embraces feature
warping rather than image warping as practiced in FlowNet2.
Our code and trained models are available at
https://github.com/twhui/LiteFlowNet.

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
            <th>LiteFlowNet-pre-M6S6</th>
            <th>Flying Chairs</th>
            <th>4.43</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet/liteflownet_pre_M6S6_8x1_flyingchairs_320x448.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet/liteflownet_pre_M6S6_8x1_flyingchairs_320x448.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet/liteflownet_pre_M6S6_8x1_flyingchairs_320x448.pth'>Model</a></th>
        </tr>
        <tr>
            <th>LiteFlowNet-pre-M6S6R6</th>
            <th>Flying Chairs</th>
            <th>4.07</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet/liteflownet_pre_M6S6R6_8x1_flyingchairs_320x448.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet/liteflownet_pre_M6S6R6_8x1_flyingchairs_320x448.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet/liteflownet_pre_M6S6R6_8x1_flyingchairs_320x448.pth'>Model</a></th>
        </tr>
        <tr>
            <th>LiteFlowNet-pre-M5S5R5</th>
            <th>Flying Chairs</th>
            <th>2.98</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet/liteflownet_pre_M5S5R5_8x1_flyingchairs_320x448.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet/liteflownet_pre_M5S5R5_8x1_flyingchairs_320x448.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet/liteflownet_pre_M5S5R5_8x1_flyingchairs_320x448.pth'>Model</a></th>
        </tr>
        <tr>
            <th>LiteFlowNet-pre-M4S4R4</th>
            <th>Flying Chairs</th>
            <th>2.20</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet/liteflownet_pre_M4S4R4_8x1_flyingchairs_320x448.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet/liteflownet_pre_M4S4R4_8x1_flyingchairs_320x448.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet/liteflownet_pre_M4S4R4_8x1_flyingchairs_320x448.pth'>Model</a></th>
        </tr>
        <tr>
            <th>LiteFlowNet-pre-M3S3R3</th>
            <th>Flying Chairs</th>
            <th>1.71</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet/liteflownet_pre_M3S3R3_8x1_flyingchairs_320x448.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet/liteflownet_pre_M3S3R3_8x1_flyingchairs_320x448.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet/liteflownet_pre_M3S3R3_8x1_flyingchairs_320x448.pth'>Model</a></th>
        </tr>
        <tr>
            <th>LiteFlowNet-pre (LiteFlowNet-pre-M2S2R2)</th>
            <th>Flying Chairs</th>
            <th>1.38</th>
            <th>2.74</th>
            <th>4.52</th>
            <th>6.49</th>
            <th>37.99%</th>
            <th>15.41</th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet/liteflownet_pre_M2S2R2_8x1_flyingchairs_320x448.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet/liteflownet_pre_M2S2R2_8x1_flyingchairs_320x448.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet/liteflownet_pre_M2S2R2_8x1_flyingchairs_320x448.pth'>Model</a></th>
        </tr>
        <tr>
            <th>LiteFlowNet</th>
            <th>Flying Chairs + Flying Thing3d subset</th>
            <th>-</th>
            <th>2.47</th>
            <th>4.30</th>
            <th>5.42</th>
            <th>32.86$</th>
            <th>13.50</th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet/liteflownet_8x1_500k_flyingthings3d_subset_384x768.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet/liteflownet_8x1_500k_flyingthings3d_subset_384x768.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet/liteflownet_8x1_500k_flyingthings3d_subset_384x768.pth'>Model</a></th>
        </tr>
        <tr>
            <th>LiteFlowNet-ft</th>
            <th>Flying Chairs + Flying Thing3d subset + Sintel</th>
            <th>-</th>
            <th>1.47</th>
            <th>2.06</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet/liteflownet_ft_4x1_500k_sintel_384x768.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet/liteflownet_ft_4x1_500k_sintel_384x768.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet/liteflownet_ft_4x1_500k_sintel_384x768.pth'>Model</a></th>
        </tr>
        <tr>
            <th>LiteFlowNet-ft</th>
            <th>Flying Chairs + Flying Thing3d subset + KITTI</th>
            <th>-</th>
            <th></th>
            <th></th>
            <th>1.07</th>
            <th>5.45%</th>
            <th>1.45</th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet/liteflownet_ft_4x1_500k_kitti_320x896.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet/liteflownet_ft_4x1_500k_kitti_320x896.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/liteflownet/liteflownet_ft_4x1_500k_kitti_320x896.pth'>Model</a></th>
        </tr>
    </tbody>
</table>

## Citation

```bibtex
@inproceedings{hui2018liteflownet,
  title={Liteflownet: A lightweight convolutional neural network for optical flow estimation},
  author={Hui, Tak-Wai and Tang, Xiaoou and Loy, Chen Change},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={8981--8989},
  year={2018}
}
```
