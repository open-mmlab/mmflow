# FlowNet2

## Introduction

<!-- [ALGORITHM] -->

```bibtex
@inproceedings{ilg2017flownet,
  title={Flownet 2.0: Evolution of optical flow estimation with deep networks},
  author={Ilg, Eddy and Mayer, Nikolaus and Saikia, Tonmoy and Keuper, Margret and Dosovitskiy, Alexey and Brox, Thomas},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2462--2470},
  year={2017}
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
