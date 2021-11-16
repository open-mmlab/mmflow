<div align="center">
  <img src="resources/mmflow-logo.png" width="600"/>
</div>

[![PyPI](https://img.shields.io/pypi/v/mmflow)](https://pypi.org/project/mmflow)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmflow.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmflow/workflows/build/badge.svg)](https://github.com/open-mmlab/mmflow/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmflow/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmflow)
[![license](https://img.shields.io/github/license/open-mmlab/mmflow.svg)](https://github.com/open-mmlab/mmflow/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmflow.svg)](https://github.com/open-mmlab/mmflow/issues)

Documentation: https://mmflow.readthedocs.io/

## Introduction

English | [简体中文](README_zh-CN.md)

MMFlow is an open source optical flow toolbox based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

The master branch works with **PyTorch 1.5+**.

![demo image](resources/sintel_mountain_1.gif)

### Major features

- **The First Unified Framework for Optical Flow**

  MMFlow is the first toolbox that provides a framework for unified implementation and evaluation of optical flow methods.

- **Flexible and Modular Design**

  We decompose the flow estimation framework into different components and one can build a new model easily and flexibly by combining different modules.

- **Plenty of Algorithms and Datasets Out of the Box**

  The toolbox directly supports popular and contemporary optical flow methods, *e.g.* FlowNet, PWC-Net, RAFT, etc,
  and representative datasets, FlyingChairs, FlyingThings3D, Sintel, KITTI, etc.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Benchmark and model zoo

Results and models are available in the [model zoo](docs/model_zoo.md).

Supported methods:

- [x] [FlowNet (ICCV'2015)](configs/flownet/README.md)
- [x] [FlowNet2 (CVPR'2017)](configs/flownet2/README.md)
- [x] [PWC-Net (CVPR'2018)](configs/pwcnet/README.md)
- [x] [LiteFlowNet (CVPR'2018)](config/liteflownet/README.md)
- [x] [LiteFlowNet2 (TPAMI'2020)](config/liteflownet2/README.md)
- [x] [IRR (CVPR'2019)](config/irr/README.md)
- [x] [MaskFlownet (CVPR'2020)](config/maskflownet/README.md)
- [x] [RAFT (ECCV'2020)](config/raft/README.md)

## Installation

Please refer to [install.md](docs/install.md) for installation and
guidance in [dataset_prepare](docs/dataset_prepare.md) for dataset preparation.

## Getting Started

If you're new of optical flow, you can start with [Learn the basics](docs/intro.md). If you’re familiar with it, check out [getting_started.md](docs/getting_started.md) to try out MMFlow.

Refer to the below tutorials to dive deeper:

- [config](docs/tutorials/0_config.md)

- [model inference](docs/tutorials/1_inference.md)

- [fine tuning](docs/tutorials/2_finetune.md)

- [data pipeline](docs/tutorials/3_data_pipeline.md)

- [add new modules](docs/tutorials/4_new_modules.md)

- [customized runtime](docs/tutorials/5_customize_runtime.md)

## Contributing

We appreciate all contributions improving MMFlow. Please refer to [CONTRIBUTING.md](https://github.com/open-mmlab/mmcv/blob/master/CONTRIBUTING.md) in MMCV for more details about the contributing guideline.

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

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM Installs OpenMMLab Packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMOCR](https://github.com/open-mmlab/mmocr): A Comprehensive Toolbox for Text Detection, Recognition and Understanding.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
