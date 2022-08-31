<div align="center">
  <img src="resources/mmflow-logo.png" width="600"/>
    <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab website</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab platform</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>
</div>

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mmflow)](https://pypi.org/project/mmflow/)
[![PyPI](https://img.shields.io/pypi/v/mmflow)](https://pypi.org/project/mmflow)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmflow.readthedocs.io/en/1.x/)
[![badge](https://github.com/open-mmlab/mmflow/workflows/build/badge.svg)](https://github.com/open-mmlab/mmflow/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmflow/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmflow)
[![license](https://img.shields.io/github/license/open-mmlab/mmflow.svg)](https://github.com/open-mmlab/mmflow/blob/1.x/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmflow.svg)](https://github.com/open-mmlab/mmflow/issues)

Documentation: <https://mmflow.readthedocs.io/en/1.x>

## Introduction

English | [简体中文](README_zh-CN.md)

MMFlow is an open source optical flow toolbox based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

The 1.x branch works with **PyTorch 1.6+**.

<https://user-images.githubusercontent.com/76149310/141947796-af4f1e67-60c9-48ed-9dd6-fcd809a7d991.mp4>

### Major features

- **The First Unified Framework for Optical Flow**

  MMFlow is the first toolbox that provides a framework for unified implementation and evaluation of optical flow algorithms.

- **Flexible and Modular Design**

  We decompose the flow estimation framework into different components,
  which makes it much easy and flexible to build a new model by combining different modules.

- **Plenty of Algorithms and Datasets Out of the Box**

  The toolbox directly supports popular and contemporary optical flow models, *e.g.* FlowNet, PWC-Net, RAFT, etc,
  and representative datasets, FlyingChairs, FlyingThings3D, Sintel, KITTI, etc.

## What's New

v1.0.0rc0 was released in 31/8/2022.
Please refer to [changelog.md](docs/en/notes/changelog.md) for details and release history.

- Unifies interfaces of all components based on MMEngine.
- Faster training and testing speed with complete support of mixed precision training.
- Refactored and more flexible architecture.

## Installation

Please refer to [install.md](docs/en/install.md) for installation and
guidance in [dataset_prepare](docs/en/user_guides/2_dataset_prepare.md) for dataset preparation.

## Get Started

Please see [Overview](docs/en/overview.md) for the general introduction of MMFlow.

Please see [user guides](https://mmflow.readthedocs.io/en/1.x/user_guides/index.html) for the basic usage of MMFlow.
There are also [advanced tutorials](https://mmflow.readthedocs.io/en/1.x/advanced_guides/index.html) for in-depth understanding of mmflow design and implementation .

To migrate from MMFlow 0.x, please refer to [migration](docs/en/migration.md).

## Benchmark and model zoo

Results and models are available in the [model zoo](docs/en/model_zoo.md).

Supported methods:

- [x] [FlowNet (ICCV'2015)](configs/flownet/README.md)
- [x] [FlowNet2 (CVPR'2017)](configs/flownet2/README.md)
- [x] [PWC-Net (CVPR'2018)](configs/pwcnet/README.md)
- [x] [LiteFlowNet (CVPR'2018)](configs/liteflownet/README.md)
- [x] [LiteFlowNet2 (TPAMI'2020)](configs/liteflownet2/README.md)
- [x] [IRR (CVPR'2019)](configs/irr/README.md)
- [x] [MaskFlownet (CVPR'2020)](configs/maskflownet/README.md)
- [x] [RAFT (ECCV'2020)](configs/raft/README.md)
- [x] [GMA (ICCV' 2021)](configs/gma/README.md)

## Contributing

We appreciate all contributions improving MMFlow. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for more details about the contributing guideline.

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```BibTeX
@misc{2021mmflow,
    title={{MMFlow}: OpenMMLab Optical Flow Toolbox and Benchmark},
    author={MMFlow Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmflow}},
    year={2021}
}
```

## Projects in OpenMMLab

- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab foundational library for training deep learning models
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition, and understanding toolbox.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning toolbox and benchmark.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab model compression toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab fewshot learning toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab Model Deployment Framework.
