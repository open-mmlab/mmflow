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

文档: https://mmflow.readthedocs.io/

[English](README.md) | 简体中文

## 简介

MMFlow 是一款基于 PyTorch 的光流工具箱，是 [OpenMMLab](http://openmmlab.org/) 项目的成员之一。

主分支代码目前支持 **PyTorch 1.5 以上**的版本。

https://user-images.githubusercontent.com/76149310/141947796-af4f1e67-60c9-48ed-9dd6-fcd809a7d991.mp4

### 主要特性

- **首个光流算法的统一框架**

  MMFlow 是第一个提供光流方法统一实现和评估框架的工具箱。

- **模块化设计**

  MMFlow 将光流估计框架解耦成不同的模块组件，通过组合不同的模块组件，用户可以便捷地构建自定义的光流算法模型。

- **丰富的开箱即用的算法和数据集**

  MMFlow 支持了众多主流经典的光流算法，例如 FlowNet, PWC-Net, RAFT 等，
  以及多种数据集的准备和构建，如 FlyingChairs, FlyingThings3D, Sintel, KITTI 等。

## 开源许可证

该项目采用 [Apache 2.0 开源许可证](LICENSE)。

## 基准测试和模型库

测试结果和模型可以在[模型库](docs/model_zoo.md)中找到。

已支持的算法：

- [x] [FlowNet (ICCV'2015)](configs/flownet/README.md)
- [x] [FlowNet2 (CVPR'2017)](configs/flownet2/README.md)
- [x] [PWC-Net (CVPR'2018)](configs/pwcnet/README.md)
- [x] [LiteFlowNet (CVPR'2018)](configs/liteflownet/README.md)
- [x] [LiteFlowNet2 (TPAMI'2020)](configs/liteflownet2/README.md)
- [x] [IRR (CVPR'2019)](configs/irr/README.md)
- [x] [MaskFlownet (CVPR'2020)](configs/maskflownet/README.md)
- [x] [RAFT (ECCV'2020)](configs/raft/README.md)


## 安装

请参考[安装文档](docs/install.md)进行安装, 参考[数据准备](docs/dataset_prepare.md)准备数据集。

## 快速入门

如果初次解除光流算法，你可以从[Learn the basics](docs/intro.md)开始了解光流的基本概念和 MMFlow 的框架。
如果对光流很熟悉，请参考[getting_started.md](docs/getting_started.md)上手使用 MMFlow.

MMFlow 也提供了其他更详细的教程，包括：

- [配置文件](docs/tutorials/0_config.md)

- [模型推理](docs/tutorials/1_inference.md)

- [微调模型](docs/tutorials/2_finetune.md)

- [数据预处理](docs/tutorials/3_data_pipeline.md)

- [添加新模型](docs/tutorials/4_new_modules.md)

- [自定义模型运行参数](docs/tutorials/6_customize_runtime.md)。

## 贡献指南

我们感谢所有的贡献者为改进和提升 MMFlow 所作出的努力。请参考[贡献指南](https://github.com/open-mmlab/mmcv/blob/master/CONTRIBUTING.md)来了解参与项目贡献的相关指引。

## 引用

如果您发现此项目对您的研究有用，请考虑引用：

```BibTeX
@misc{2021mmflow,
    title={{MMFlow}: OpenMMLab Optical Flow Toolbox and Benchmark},
    author={MMFlow Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmflow}},
    year={2021}
}
```

## OpenMMLab 的其他项目

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库
- [MIM](https://github.com/open-mmlab/mim): MIM 是 OpenMMLab 项目、算法、模型的统一入口
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab 图像分类工具箱与测试基准
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 检测工具箱与测试基准
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab 新一代通用3D目标检测平台
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab 语义分割工具箱与测试基准
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab 新一代视频理解工具箱与测试基准
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab 一体化视频目标感知平台
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab 姿态估计工具箱与测试基准
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab 图像视频编辑工具箱
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab 全流程文字检测识别理解工具包
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab 新一代生成模型工具箱
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab 光流估计工具箱与测试基准

## 欢迎加入 OpenMMLab 社区

 扫描下方的二维码可关注 OpenMMLab 团队的 [知乎官方账号](https://www.zhihu.com/people/openmmlab)，加入 OpenMMLab 团队的 [官方交流 QQ 群](https://jq.qq.com/?_wv=1027&k=aCvMxdr3)

<div align="center">
<img src="resources/zhihu_qrcode.jpg" height="400" />  <img src="resources/qq_group_qrcode.jpg" height="400" />
</div>

我们会在 OpenMMLab 社区为大家

- 📢 分享 AI 框架的前沿核心技术
- 💻 解读 PyTorch 常用模块源码
- 📰 发布 OpenMMLab 的相关新闻
- 🚀 介绍 OpenMMLab 开发的前沿算法
- 🏃 获取更高效的问题答疑和意见反馈
- 🔥 提供与各行各业开发者充分交流的平台

干货满满 📘，等你来撩 💗，OpenMMLab 社区期待您的加入 👬
