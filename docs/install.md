# Installation

<!-- TOC -->

- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Prepare environment](#prepare-environment)
  - [Install MMFlow](#install-mmflow)
  - [A from-scratch setup script](#a-from-scratch-setup-script)
  - [Verification](#verification)

<!-- TOC -->

## Prerequisites

- Linux
- Python 3.6+
- PyTorch 1.5 or higher
- CUDA 9.0 or higher
- NCCL 2
- GCC 5.4 or higher
- [mmcv](https://github.com/open-mmlab/mmcv) 1.3.15 or higher

## Prepare environment

a. Create a conda virtual environment and activate it.

```shell
conda create -n openmmlab python=3.7 -y
conda activate openmmlab
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/)

Note: Make sure that your compilation CUDA version and runtime CUDA version match.
You can check the supported CUDA version for pre-compiled packages on the [PyTorch website](https://pytorch.org/).

`E.g.1` If you have CUDA 10.2 installed under `/usr/local/cuda` and would like to install the latest PyTorch,
you can run this command.

```shell
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

`E.g.2` If you have CUDA 9.2 installed under `/usr/local/cuda` and would like to install PyTorch 1.7.0.,
you can run this command.

```shell
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=9.2 -c pytorch
```

If you build PyTorch from source instead of installing the pre-built package, you can use more CUDA versions such as 9.0.

c. Install MMCV, we recommend you to install the pre-built mmcv as below.

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```

Please replace ``{cu_version}`` and ``{torch_version}`` in the url to your desired one. For example, to install the latest ``mmcv-full`` with ``CUDA 10.2`` and ``PyTorch 1.10.0``, use the following command:

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html
```

See [here](https://github.com/open-mmlab/mmcv#installation) for different versions of MMCV compatible to different PyTorch and CUDA versions.

Optionally you can choose to compile mmcv from source by the following command

```shell
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
MMCV_WITH_OPS=1 pip install -e .  # package mmcv-full, which contains cuda ops, will be installed after this step
# OR pip install -e .  # package mmcv, which contains no cuda ops, will be installed after this step
cd ..
```

**Important:** You need to run `pip uninstall mmcv` first if you have mmcv installed. If `mmcv` and `mmcv-full` are both installed, there will be `ModuleNotFoundError`.
## Install MMFlow

a. Clone the MMFlow repository.

```shell
git clone https://github.com/open-mmlab/mmflow.git
cd mmflow
```

b. Install build requirements and then install mmflow.

```shell
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```

Note:

1. The git commit id will be written to the version number, e.g. 0.6.0+2e7045c. The version will also be saved in trained models.

2. Following the above instructions, MMFlow is installed on `dev` mode, any local modifications made to the code will take effect without the need to reinstall it (unless you submit some commits and want to update the version number).

3. If you would like to use `opencv-python-headless` instead of `opencv-python`, you can install it before installing MMCV.

## A from-scratch setup script

Assuming that you already have CUDA 10.1 installed, here is a full script for setting up mmflow with conda.

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch -y

# install latest mmcv
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html

# install mmflow
git clone https://github.com/open-mmlab/mmflow.git
cd mmflow
pip install -r requirements/build.txt
pip install -v -e .
```

## Verification

To verify whether MMFlow is installed correctly, we can run the following sample code to initialize a model and inference a demo image.

```python
from mmflow.apis import inference_model, init_model

config_file = 'configs/pwcnet/pwcnet_ft_4x1_300k_sintel_final_384x768.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmflow/pwcnet/pwcnet_ft_4x1_300k_sintel_final_384x768.pth
checkpoint_file = 'checkpoints/pwcnet_ft_4x1_300k_sintel_final_384x768.pth'
device = 'cuda:0'
# init a model
model = init_model(config_file, checkpoint_file, device=device)
# inference the demo image
inference_model(model, 'demo/frame_0001.png', 'demo/frame_0002.png')
```

The above code is supposed to run successfully upon you finish the installation.
