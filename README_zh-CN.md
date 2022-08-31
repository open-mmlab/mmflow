<div align="center">
  <img src="resources/mmflow-logo.png" width="600"/>
    <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab 官网</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab 开放平台</font></b>
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
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmflow.readthedocs.io/en/dev-1.x/)
[![badge](https://github.com/open-mmlab/mmflow/workflows/build/badge.svg)](https://github.com/open-mmlab/mmflow/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmflow/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmflow)
[![license](https://img.shields.io/github/license/open-mmlab/mmflow.svg)](https://github.com/open-mmlab/mmflow/blob/dev-1.x/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmflow.svg)](https://github.com/open-mmlab/mmflow/issues)

文档: <https://mmflow.readthedocs.io/en/dev-1.x>

[English](README.md) | 简体中文

## 简介

MMFlow 是一款基于 PyTorch 的光流工具箱，是 [OpenMMLab](http://openmmlab.org/) 项目的成员之一。

dev-1.x 分支代码目前支持 **PyTorch 1.6 以上**的版本。

<https://user-images.githubusercontent.com/76149310/141947796-af4f1e67-60c9-48ed-9dd6-fcd809a7d991.mp4>

### 主要特性

- **首个光流算法的统一框架**

  MMFlow 是第一个提供光流方法统一实现和评估框架的工具箱。

- **模块化设计**

  MMFlow 将光流估计框架解耦成不同的模块组件，通过组合不同的模块组件，用户可以便捷地构建自定义的光流算法模型。

- **丰富的开箱即用的算法和数据集**

  MMFlow 支持了众多主流经典的光流算法，例如 FlowNet, PWC-Net, RAFT 等，
  以及多种数据集的准备和构建，如 FlyingChairs, FlyingThings3D, Sintel, KITTI 等。

## 更新日志

最新版本 v1.0.0rc0 在 2022.8.31 发布。
如果想了解更多版本更新细节和历史信息，请阅读[更新日志](docs/en/notes/changelog.md)。

如果想了解更多版本更新细节和历史信息，请参考[更新日志](docs/en/changelog.md)。

## 基准测试和模型库

测试结果和模型可以在[模型库](docs/zh_cn/model_zoo.md)中找到。

已支持的算法：

- [x] [FlowNet (ICCV'2015)](configs/flownet/README.md)
- [x] [FlowNet2 (CVPR'2017)](configs/flownet2/README.md)
- [x] [PWC-Net (CVPR'2018)](configs/pwcnet/README.md)
- [x] [LiteFlowNet (CVPR'2018)](configs/liteflownet/README.md)
- [x] [LiteFlowNet2 (TPAMI'2020)](configs/liteflownet2/README.md)
- [x] [IRR (CVPR'2019)](configs/irr/README.md)
- [x] [MaskFlownet (CVPR'2020)](configs/maskflownet/README.md)
- [x] [RAFT (ECCV'2020)](configs/raft/README.md)
- [x] [GMA (ICCV' 2021)](configs/gma/README.md)

## 安装

请参考[安装文档](docs/en/install.md)进行安装, 参考[数据准备](docs/en/user_guides/2_dataset_prepare.md)准备数据集。

## 快速入门

请参考[概述](docs/zh_cn/overview.md)对 MMFlow 进行初步了解

请参考[用户指南](https://mmflow.readthedocs.io/zh_CN/dev-1.x/user_guides/index.html)了解 mmflow 的基本使用，以及[进阶指南](https://mmflow.readthedocs.io/zh_CN/dev-1.x/advanced_guides/index.html)深入了解 mmflow 设计和代码实现。

若需要将 0.x 版本的代码迁移至新版，请参考[迁移文档](docs/zh_cn/migration.md)

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
- [MIM](https://github.com/open-mmlab/mim): MIM 是 OpenMMlab 项目、算法、模型的统一入口
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab 图像分类工具箱
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 目标检测工具箱
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab 新一代通用 3D 目标检测平台
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab 旋转框检测工具箱与测试基准
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab 语义分割工具箱
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab 全流程文字检测识别理解工具包
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab 姿态估计工具箱
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 人体参数化模型工具箱与测试基准
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab 自监督学习工具箱与测试基准
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab 模型压缩工具箱与测试基准
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab 少样本学习工具箱与测试基准
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab 新一代视频理解工具箱
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab 一体化视频目标感知平台
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab 光流估计工具箱与测试基准
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab 图像视频编辑工具箱
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab 图片视频生成模型工具箱
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab 模型部署框架

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
