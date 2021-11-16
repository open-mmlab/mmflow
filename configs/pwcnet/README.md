# PWC-Net

## Introduction

<!-- [ALGORITHM] -->

```bibtex
@inproceedings{sun2018pwc,
  title={Pwc-net: Cnns for optical flow using pyramid, warping, and cost volume},
  author={Sun, Deqing and Yang, Xiaodong and Liu, Ming-Yu and Kautz, Jan},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={8934--8943},
  year={2018}
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
            <th>PWC-Net</th>
            <th>FlyingChairs</th>
            <th>1.51</th>
            <th>3.52</th>
            <th>4.81</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th><a href='https://download.openmmlab.com/mmflow/pwcnet/pwcnet_8x1_slong_flyingchairs_384x448.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/pwcnet/pwcnet_8x1_slong_flyingchairs_384x448.py'>config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/pwcnet/pwcnet_8x1_slong_flyingchairs_384x448.pth'>model</a></th>
        </tr>
        <tr>
            <th>PWC-Net</th>
            <th>Flying Chairs + FlyingThing3d subset</th>
            <th>-</th>
            <th>2.26</th>
            <th>3.79</th>
            <th>3.66</th>
            <th>29.85%</th>
            <th>9.49</th>
            <th><a href='https://download.openmmlab.com/mmflow/pwcnet/pwcnet_8x1_sfine_flyingthings3d_subset_384x768.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/pwcnet/pwcnet_8x1_sfine_flyingthings3d_subset_384x768.py'>config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/pwcnet/pwcnet_8x1_sfine_flyingthings3d_subset_384x768.pth'>model</a></th>
        </tr>
        <tr>
            <th>PWC-Net-ft</th>
            <th>Flying Chairs + FlyingThing3d subset + Sintel</th>
            <th>-</th>
            <th>1.50</th>
            <th>2.06</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th><a href='https://download.openmmlab.com/mmflow/pwcnet/pwcnet_ft_4x1_300k_sintel_384x768.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/pwcnet/pwcnet_ft_4x1_300k_sintel_384x768.py'>config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/pwcnet/pwcnet_ft_4x1_300k_sintel_384x768.pth'>model</a></th>
        </tr>
        <tr>
            <th>PWC-Net-ft-final</th>
            <th>FlyingChairs + FlyingThing3d subset + Sintel final</th>
            <th>-</th>
            <th>1.82</th>
            <th>1.78</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th><a href='https://download.openmmlab.com/mmflow/pwcnet/pwcnet_ft_4x1_300k_sintel_final_384x768.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/pwcnet/pwcnet_ft_4x1_300k_sintel_final_384x768.py'>config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/pwcnet/pwcnet_ft_4x1_300k_sintel_final_384x768.pth'>model</a></th>
        </tr>
        <tr>
            <th>PWC-Net-ft</th>
            <th>FlyingChairs + FlyingThing3d subset + KITTI</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th>1.07</th>
            <th>6.09%</th>
            <th>1.64</th>
            <th><a href='https://download.openmmlab.com/mmflow/pwcnet/pwcnet_ft_4x1_300k_kitti_320x896.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/pwcnet/pwcnet_ft_4x1_300k_kitti_320x896.py'>config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/pwcnet/pwcnet_ft_4x1_300k_kitti_320x896.pth'>model</a></th>
        </tr>
        <tr>
            <th>PWC-Net+</th>
            <th>FlyingChairs + FlyingThing3d subset + Sintel + KITTI2015 + HD1K</th>
            <th>-</th>
            <th>1.90</th>
            <th>2.39</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th><a href='https://download.openmmlab.com/mmflow/pwcnet/pwcnet_plus_8x1_750k_sintel_kitti2015_hd1k_320x768.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/pwcnet/pwcnet_plus_8x1_750k_sintel_kitti2015_hd1k_320x768.py'>config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/pwcnet/pwcnet_plus_8x1_750k_sintel_kitti2015_hd1k_320x768.pth'>model</a></th>
        </tr>
    </tbody>
</table>
