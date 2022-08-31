# Get Started: Install and Run MMFlow

## Prerequisites

In this section we demonstrate how to prepare an environment with PyTorch.

MMFlow works on Linux, Windows and macOS. It requires Python 3.6+, CUDA 9.2+ and PyTorch 1.5+.

**Note:**
If you are experienced with PyTorch and have already installed it, just skip this part and jump to the [next section](##installation). Otherwise, you can follow these steps for the preparation.

**Step 0.** Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 1.** Create a conda environment and activate it.

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

On GPU platforms:

```shell
conda install pytorch torchvision -c pytorch
```

On CPU platforms:

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

## Installation

We recommend that users follow our best practices to install MMFlow. However, the whole process is highly customizable. See [Customize Installation](#customize-installation) section for more information.

### Best Practices

**Step 0.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install 'mmcv>=2.0.0rc1'
mim install mmmenigne
```

**Step 1.** Install MMFlow.

Case a: If you develop and run mmflow directly, install it from source:

```shell
git clone https://github.com/open-mmlab/mmflow.git
cd mmflow
git checkout dev-1.x
# branch 'dev-1.x' set up to track remote branch 'dev-1.x' from 'origin'.
pip install -v -e .
# '-v' means verbose, or more output
# '-e' means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

Case b: If you use mmflow as a dependency or third-party package, install it with pip:

```shell
pip install 'mmflow>=1.0.0rc0'
```

## Verify the installation

To verify whether MMFlow is installed correctly, we provide some sample codes to run an inference demo.

**Step 1.** We need to download config and checkpoint files.

```shell
mim download mmflow --config pwcnet-ft_4xb1_300k_sintel-final-384x768.py
```

The downloading will take several seconds or more, depending on your network environment. When it is done, you will find two files
`pwcnet-ft_4xb1_300k_sintel-final-384x768.py` and `pwcnet_ft_4x1_300k_sintel_final_384x768.pth` in your current folder.

**Step 2.** Verify the inference demo.

Option (a). If you install mmflow from source, just run the following command.

```shell
   python demo/image_demo.py demo/frame_0001.png demo/frame_0002.png \
       configs/pwcnet/pwcnet_ft_4x1_300k_sintel_final_384x768.py \
       checkpoints/pwcnet_ft_4x1_300k_sintel_final_384x768.pth results
```

Output will be saved in the directory `results` including a rendered flow map `flow.png` and flow file `flow.flo`

Option (b). If you install mmflow with pip, open you python interpreter and copy&paste the following codes.

```python
from mmflow.apis import inference_model, init_model
from mmflow.utils import register_all_modules

register_all_modules()
config_file = 'pwcnet-ft_4xb1_300k_sintel-final-384x768.py'
checkpoint_file = 'pwcnet_ft_4x1_300k_sintel_final_384x768.pth'
device = 'cuda:0'
# init a model
model = init_model(config_file, checkpoint_file, device=device)
# inference the demo image
inference_model(model, 'demo/frame_0001.png', 'demo/frame_0002.png')
```

You will see a array printed, which is the flow data.

## Customize Installation

### CUDA versions

When installing PyTorch, you need to specify the version of CUDA. If you are not clear on which to choose, follow our recommendations:

- For Ampere-based NVIDIA GPUs, such as GeForce 30 series and NVIDIA A100, CUDA 11 is a must.
- For older NVIDIA GPUs, CUDA 11 is backward compatible, but CUDA 10.2 offers better compatibility and is more lightweight.

Please make sure the GPU driver satisfies the minimum version requirements. See [this table](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions) for more information.

```{note}
Installing CUDA runtime libraries is enough if you follow our best practices, because no CUDA code will be compiled locally. However if you hope to compile MMCV from source or develop other CUDA operators, you need to install the complete CUDA toolkit from NVIDIA's [website](https://developer.nvidia.com/cuda-downloads), and its version should match the CUDA version of PyTorch. i.e., the specified version of cudatoolkit in `conda install` command.
```

### Install MMCV without MIM

MMCV contains C++ and CUDA extensions, thus depending on PyTorch in a complex way. MIM solves such dependencies automatically and makes the installation easier. However, it is not a must.

To install MMCV with pip instead of MIM, please follow [MMCV installation guides](https://mmcv.readthedocs.io/en/latest/get_started/installation.html). This requires manually specifying a find-url based on PyTorch version and its CUDA version.

For example, the following command install mmcv-full built for PyTorch 1.10.x and CUDA 11.3.

```shell
pip install mmcv==2.0.0rc1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
```

### Install on CPU-only platforms

MMFlow can be built for CPU only environment. In CPU mode you can train (requires MMCV version >= 1.4.4), test or inference a model.

However some functionalities are gone in this mode:

- Correlation

If you try to train/test/inference a model containing above ops, an error will be raised. The following table lists affected algorithms.

|  Operator   |                                    Model                                     |
| :---------: | :--------------------------------------------------------------------------: |
| Correlation | PWC-Net, FlowNetC, FlowNet2, IRR-PWC, LiteFlowNet, LiteFlowNet2, MaskFlowNet |

### Install on Google Colab

[Google Colab](https://research.google.com/) usually has PyTorch installed,
thus we only need to install MMCV and MMFlow with the following commands.

**Step 1.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
!pip3 install openmim
!mim install mmcv>=2.0.0rc1
```

**Step 2.** Install MMFlow from the source.

```shell
!git clone https://github.com/open-mmlab/mmflow.git
%cd mmflow
!git checkout dev-1.x
!pip install -e .
```

**Step 3.** Verification.

```python
import mmflow
print(mmflow.__version__)
# Example output: 1.0.0rc0
```

```{note}
Within Jupyter, the exclamation mark `!` is used to call external executables and `%cd` is a [magic command](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-cd) to change the current working directory of Python.
```

### Using MMFlow with Docker

We provide a [Dockerfile](https://github.com/open-mmlab/mmflow/blob/master/docker/Dockerfile) to build an image. Ensure that your [docker version](https://docs.docker.com/engine/install/) >=19.03.

```shell
# build an image with PyTorch 1.6, CUDA 10.1
# If you prefer other versions, just modified the Dockerfile
docker build -t mmflow docker/
```

Run it with

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmflow/data mmflow
```

## Trouble shooting

If you have some issues during the installation, please first view the [FAQ](faq.md) page.
You may [open an issue](https://github.com/open-mmlab/mmflow/issues/new/choose) on GitHub if no solution is found.
