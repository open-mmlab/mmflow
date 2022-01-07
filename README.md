<div align="center">
  <img src="resources/mmflow-logo.png" width="600"/>
</div>

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mmflow)](https://pypi.org/project/mmflow/)
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

https://user-images.githubusercontent.com/76149310/141947796-af4f1e67-60c9-48ed-9dd6-fcd809a7d991.mp4

### Major features

- **The First Unified Framework for Optical Flow**

  MMFlow is the first toolbox that provides a framework for unified implementation and evaluation of optical flow algorithms.

- **Flexible and Modular Design**

  We decompose the flow estimation framework into different components,
  which makes it much easy and flexible to build a new model by combining different modules.

- **Plenty of Algorithms and Datasets Out of the Box**

  The toolbox directly supports popular and contemporary optical flow models, *e.g.* FlowNet, PWC-Net, RAFT, etc,
  and representative datasets, FlyingChairs, FlyingThings3D, Sintel, KITTI, etc.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Changelog

**0.2.0** was released in 07/01/2022:

- Support [GMA](../../configs/gma/README.md): Learning to Estimate Hidden Motions with Global Motion Aggregation (ICCV 2021)
- Fix the bug of wrong refine iter in RAFT, and update [RAFT](../../configs/raft/README.md) model checkpoint after the bug fixing
- Support resuming from the latest checkpoint automatically

Please refer to [changelog.md](docs/en/changelog.md) for details and release history.

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

## Installation

Please refer to [install.md](docs/en/install.md) for installation and
guidance in [dataset_prepare](docs/en/dataset_prepare.md) for dataset preparation.

## Getting Started

If you're new of optical flow, you can start with [learn the basics](docs/en/intro.md). If you’re familiar with it, check out [getting_started](docs/en/getting_started.md) to try out MMFlow.

Refer to the below tutorials to dive deeper:

- [config](docs/en/tutorials/0_config.md)

- [model inference](docs/en/tutorials/1_inference.md)

- [fine tuning](docs/en/tutorials/2_finetune.md)

- [data pipeline](docs/en/tutorials/3_data_pipeline.md)

- [add new modules](docs/en/tutorials/4_new_modules.md)

- [customized runtime](docs/en/tutorials/5_customize_runtime.md)

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
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab FewShot Learning Toolbox and Benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab Human Pose and Shape Estimation Toolbox and Benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning toolbox and benchmark.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab Model Compression Toolbox and Benchmark.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab Model Deployment Framework.
