# GMA

[Learning to Estimate Hidden Motions with Global Motion Aggregation](https://arxiv.org/pdf/2104.02409.pdf)

<!-- [ALGORITHM] -->

## Abstract

Occlusions pose a significant challenge to optical flow
algorithms that rely on local evidences. We consider an occluded point to be
one that is imaged in the reference frame but not in the next, a slight
overloading of the standard definition since it also includes points that move
out-of-frame.
Estimating the motion of these points is extremely difficult,
particularly in the two-frame setting. Previous work relies on CNNs
to learn occlusions, without much success, or requires multiple frames
to reason about occlusions using temporal smoothness.
In this paper, we argue that the occlusion problem can be better solved
in the two-frame case by modelling image self-similarities. We introduce
a global motion aggregation module, a transformer-based
approach to find long-range dependencies between pixels
in the first image, and perform global aggregation on the
corresponding motion features. We demonstrate that the
optical flow estimates in the occluded regions can be significantly
improved without damaging the performance in
non-occluded regions. This approach obtains new state-of-
the-art results on the challenging Sintel dataset, improving
the average end-point error by 13.6% on Sintel Final
and 13.7% on Sintel Clean. At the time of submission,
our method ranks first on these benchmarks among all published and
unpublished approaches. Code is available at https://github.com/zacjiang/GMA.

<div align=center>
<img src="https://user-images.githubusercontent.com/76149310/148016891-b8aec62a-3722-4693-913f-10181c946273.png" width="100%"/>
</div>

## Results and Models

<table>
    <thead>
        <tr>
            <td rowspan=2>Models</td>
            <td rowspan=2>Training datasets</td>
            <td rowspan=2>Flying Chairs</td>
            <td colspan=2>Sintel(training)</td>
            <td colspan=2>KITTI2015(training)</td>
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
            <th>GMA</th>
            <th>Flying Chairs</th>
            <th>0.72</th>
            <th>2.40</th>
            <th>4.53</th>
            <th>-</th>
            <th>-</th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_8x2_120k_flyingchairs_368x496.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_8x2_120k_flyingchairs_368x496.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_8x2_120k_flyingchairs_368x496.pth'>Model</a></th>
        </tr>
        <tr>
            <th>GMA</th>
            <th>FlyingChairs + FlyingThing3d</th>
            <th>-</th>
            <th>1.31</th>
            <th>2.61</th>
            <th>16.54%</th>
            <th>5.00</th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_8x2_120k_flyingthings3d_400x720.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_8x2_120k_flyingthings3d_400x720.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_8x2_120k_flyingthings3d_400x720.pth'>Model</a></th>
        </tr>
        <tr>
            <th>GMA</th>
            <th>FlyingChairs + FlyingThing3d + Sintel</th>
            <th>-</th>
            <th>0.56</th>
            <th>0.84</th>
            <th>-</th>
            <th>-</th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_8x2_120k_flyingthings3d_sintel_368x768.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_8x2_120k_flyingthings3d_sintel_368x768.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_8x2_120k_flyingthings3d_sintel_368x768.pth'>Model</a></th>
        </tr>
        <tr>
            <th>GMA</th>
            <th>Mixed Dataset<sup>[1]</sup></th>
            <th>-</th>
            <th>0.56</th>
            <th>0.85</th>
            <th>5.27%</th>
            <th>1.50</th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_8x2_120k_mixed_368x768.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_8x2_120k_mixed_368x768.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_8x2_120k_mixed_368x768.pth'>Model</a></th>
        </tr>
        <tr>
            <th>GMA</th>
            <th>KITTI2015</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th>1.34%</th>
            <th>0.58</th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_8x2_50k_kitti2015_288x960.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_8x2_50k_kitti2015_288x960.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_8x2_50k_kitti2015_288x960.pth'>Model</a></th>
        </tr>
        <tr>
            <th>GMA(p only)</th>
            <th>Flying Chairs</th>
            <th>0.76</th>
            <th>2.38</th>
            <th>4.69</th>
            <th>-</th>
            <th>-</th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_p-only_8x2_120k_flyingchairs_368x496.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_p-only_8x2_120k_flyingchairs_368x496.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_p-only_8x2_120k_flyingchairs_368x496.pth'>Model</a></th>
        </tr>
        <tr>
            <th>GMA(p only)</th>
            <th>FlyingChairs + FlyingThing3d</th>
            <th>-</th>
            <th>1.48</th>
            <th>2.73</th>
            <th>16.46%</th>
            <th>4.81</th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_p-only_8x2_120k_flyingthings3d_400x720.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_p-only_8x2_120k_flyingthings3d_400x720.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_p-only_8x2_120k_flyingthings3d_400x720.pth'>Model</a></th>
        </tr>
        <tr>
            <th>GMA(p only)</th>
            <th>Mixed Dataset<sup>[1]</sup></th>
            <th>-</th>
            <th>0.58</th>
            <th>0.89</th>
            <th>5.28%</th>
            <th>1.47</th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_p-only_8x2_120k_mixed_368x768.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_p-only_8x2_120k_mixed_368x768.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_p-only_8x2_120k_mixed_368x768.pth'>Model</a></th>
        </tr>
        <tr>
            <th>GMA(p only)</th>
            <th>KITTI2015</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th>1.39%</th>
            <th>0.58</th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_p-only_8x2_50k_kitti2015_288x960.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_p-only_8x2_50k_kitti2015_288x960.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_p-only_8x2_50k_kitti2015_288x960.pth'>Model</a></th>
        </tr>
        <tr>
            <th>GMA(+p)</th>
            <th>Flying Chairs</th>
            <th>0.73</th>
            <th>2.52</th>
            <th>4.65</th>
            <th>-</th>
            <th>-</th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_plus-p_8x2_120k_flyingchairs_368x496.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_plus-p_8x2_120k_flyingchairs_368x496.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_plus-p_8x2_120k_flyingchairs_368x496.pth'>Model</a></th>
        </tr>
        <tr>
            <th>GMA(+p)</th>
            <th>FlyingChairs + FlyingThing3d</th>
            <th>-</th>
            <th>1.38</th>
            <th>2.79</th>
            <th>19.17%</th>
            <th>6.73</th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_plus-p_8x2_120k_flyingthings3d_400x720.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_plus-p_8x2_120k_flyingthings3d_400x720.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_plus-p_8x2_120k_flyingthings3d_400x720.pth'>Model</a></th>
        </tr>
        <tr>
            <th>GMA(+p)</th>
            <th>Mixed Dataset<sup>[1]</sup></th>
            <th>-</th>
            <th>0.63</th>
            <th>0.94</th>
            <th>9.07%</th>
            <th>3.82</th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_plus-p_8x2_120k_mixed_368x768.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_plus-p_8x2_120k_mixed_368x768.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_plus-p_8x2_120k_mixed_368x768.pth'>Model</a></th>
        </tr>
        <tr>
            <th>GMA(+p)</th>
            <th>KITTI2015</th>
            <th>-</th>
            <th>-</th>
            <th>-</th>
            <th>1.50%</th>
            <th>0.62</th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_plus-p_8x2_50k_kitti2015_288x960.log.json'>log</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_plus-p_8x2_50k_kitti2015_288x960.py'>Config</a></th>
            <th><a href='https://download.openmmlab.com/mmflow/gma/gma_plus-p_8x2_50k_kitti2015_288x960.pth'>Model</a></th>
        </tr>
    </tbody>
</table>

## Citation

@article{jiang2021learning,
  title={Learning to Estimate Hidden Motions with Global Motion Aggregation},
  author={Jiang, Shihao and Campbell, Dylan and Lu, Yao and Li, Hongdong and Hartley, Richard},
  journal={arXiv preprint arXiv:2104.02409},
  year={2021}
}

[1] The mixed dataset consisted of FlyingChairs, FlyingThing3d, Sintel, KITTI2015, and HD1K.
