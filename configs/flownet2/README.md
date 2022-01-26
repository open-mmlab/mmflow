# FlowNet2

[FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks](https://openaccess.thecvf.com/content_cvpr_2017/papers/Ilg_FlowNet_2.0_Evolution_CVPR_2017_paper.pdf)

<!-- [ALGORITHM] -->

## Abstract

The FlowNet demonstrated that optical flow estimation
can be cast as a learning problem. However, the state of
the art with regard to the quality of the flow has still been
defined by traditional methods. Particularly on small displacements
and real-world data, FlowNet cannot compete with variational methods.
In this paper, we advance the concept of end-to-end learning of optical flow
and make it work really well.
The large improvements in quality and speed are caused
by three major contributions: first, we focus on the training data
and show that the schedule of presenting data during training is very important.
Second, we develop a stacked architecture that includes warping
of the second image with intermediate optical flow. Third,
we elaborate on small displacements by introducing a sub-network specializing
on small motions. FlowNet 2.0 is only marginally slower than
the original FlowNet but decreases the estimation error by more than 50%.
It performs on par with state-of-the-art methods, while running at interactive
frame rates. Moreover, we present faster variants that allow optical flow
computation at up to 140fps with accuracy matching the original FlowNet.

<div align=center>
<img src="https://user-images.githubusercontent.com/76149310/142731310-af0c4586-97b6-4a1e-9ada-50c7b2ee0851.png" width="70%"/>
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
            <th>FlowNet2CS</th>
            <th>FlyingChairs</th>
            <th>1.59</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th><a href='https://download.openmmlab.com/mmflow/flownet2/flownet2cs_8x1_slong_flyingchairs_384x448.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/flownet2/'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/flownet2/flownet2cs_8x1_slong_flyingchairs_384x448.pth'>Model</a></th>
        </tr>
        <tr>
            <th>FlowNet2CS</th>
            <th>Flying Chairs + FlyingThing3d subset</th>
            <th>-</th>
            <th>1.96</th>
            <th>3.69</th>
            <th>3.50</th>
            <th>28.28%</th>
            <th>8.23</th>
            <th><a href='https://download.openmmlab.com/mmflow/flownet2/flownet2cs_8x1_sfine_flyingthings3d_subset_384x768.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/flownet2/flownet2cs_8x1_sfine_flyingthings3d_subset_384x768.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/flownet2/flownet2cs_8x1_sfine_flyingthings3d_subset_384x768.pth'>Model</a></th>
        </tr>
        <tr>
            <th>FlowNet2CSS</th>
            <th>FlyingChairs</th>
            <th>1.55</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th><a href='https://download.openmmlab.com/mmflow/flownet2/flownet2css_8x1_slong_flyingchairs_384x448.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/flownet2/flownet2css_8x1_slong_flyingchairs_384x448.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/flownet2/flownet2css_8x1_slong_flyingchairs_384x448.pth'>Model</a></th>
        </tr>
        <tr>
            <th>FlowNet2CSS</th>
            <th>Flying Chairs + FlyingThing3d subset</th>
            <th>-</th>
            <th>1.85</th>
            <th>3.57</th>
            <th>3.13</th>
            <th>25.76%</th>
            <th>7.72</th>
            <th><a href='https://download.openmmlab.com/mmflow/flownet2/flownet2css_8x1_sfine_flyingthings3d_subset_384x768.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/flownet2/flownet2css_8x1_sfine_flyingthings3d_subset_384x768.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/flownet2/flownet2css_8x1_sfine_flyingthings3d_subset_384x768.pth'>Model</a></th>
        </tr>
        <tr>
            <th>FlowNet2CSS-sd</th>
            <th>Flying Chairs + FlyingThing3d subset + ChairsSDHom</th>
            <th>-</th>
            <th>1.81</th>
            <th>3.69</th>
            <th>2.98</th>
            <th>25.66%</th>
            <th>7.99</th>
            <th><a href='https://download.openmmlab.com/mmflow/flownet2/flownet2css-sd_8x1_sfine_flyingthings3d_subset_chairssdhom_384x448.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/flownet2/flownet2css-sd_8x1_sfine_flyingthings3d_subset_chairssdhom_384x448.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/flownet2/flownet2css-sd_8x1_sfine_flyingthings3d_subset_chairssdhom_384x448.pth'>Model</a></th>
        </tr>
        <tr>
            <th>FlowNet2</th>
            <th>FlyingThing3d subset</th>
            <th></th>
            <th>1.78</th>
            <th>3.31</th>
            <th>3.02</th>
            <th>25.18%</th>
            <th>8.02</th>
            <th><a href='https://download.openmmlab.com/mmflow/flownet2/flownet2_8x1_sfine_flyingthings3d_subset_384x768.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/flownet2/flownet2_8x1_sfine_flyingthings3d_subset_384x768.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/flownet2/flownet2_8x1_sfine_flyingthings3d_subset_384x768.pth'>Model</a></th>
        </tr>
    </tbody>
</table>

<table>
    <thead>
        <tr>
            <td rowspan=2>Models</td>
            <td rowspan=2>Training datasets</td>
            <td rowspan=2>ChairsSDHom</td>
            <td rowspan=2>Log</td>
            <td rowspan=2>Config</td>
            <td rowspan=2>Download</td>
        </tr>
    </thead>
     <tbody>
        <tr>
        <th>Flownet2sd</th>
        <th>ChairsSDHom</th>
        <th>0.37</th>
        <th><a href='https://download.openmmlab.com/mmflow/flownet2/flownet2sd_8x1_slong_chairssdhom_384x448.log.json'>log</a></th>
        <th><a href='https://download.openmmlab.com/mmflow/flownet2/flownet2sd_8x1_slong_chairssdhom_384x448.py'>Config</a></th>
        <th><a href='https://download.openmmlab.com/mmflow/flownet2/flownet2sd_8x1_slong_chairssdhom_384x448.pth'>Model</a></th>
        </tr>
    </tbody>
</table>

## Citation

```bibtex
@inproceedings{ilg2017flownet,
  title={Flownet 2.0: Evolution of optical flow estimation with deep networks},
  author={Ilg, Eddy and Mayer, Nikolaus and Saikia, Tonmoy and Keuper, Margret and Dosovitskiy, Alexey and Brox, Thomas},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2462--2470},
  year={2017}
}
```
