# Changelog of v1.x

## v1.0.0rc0 (31/8/2022)

We are excited to announce the release of MMFlow 1.0.0rc0.
MMFlow 1.0.0rc0 is a part of the OpenMMLab 2.x projects.
Built upon the new [training engine](https://github.com/open-mmlab/mmengine),
MMFlow 1.x unifies the interfaces of dataset, models, evaluation, and visualization with faster training and testing speed.

### Highlights

1. **New engines**. MMFlow 1.x is based on [MMEngine](https://github.com/open-mmlab/mmengine), which provides a general and powerful runner that allows more flexible customizations and significantly simplifies the entrypoints of high-level interfaces.

2. **Unified interfaces**. As a part of the OpenMMLab 2.x projects, MMFlow 1.x unifies and refactors the interfaces and internal logics of training, testing, datasets, models, evaluation, and visualization. All the OpenMMLab 2.x projects share the same design in those interfaces and logics to allow the emergence of multi-task/modality algorithms.

3. **Faster speed**. We optimize the training and inference speed for common models.

4. **More documentation and tutorials**. We add a bunch of documentation and tutorials to help users get started more smoothly. Read it [here](https://mmflow.readthedocs.io/en/dev-1.x/).

### Breaking Changes

We briefly list the major breaking changes here.
We will update the [migration guide](../migration.md) to provide complete details and migration instructions.

#### Dependencies

- MMFlow 1.x runs on PyTorch>=1.6. We have deprecated the support of PyTorch 1.5 to embrace the mixed precision training and other new features since PyTorch 1.6. Some models can still run on PyTorch 1.5, but the full functionality of MMFlow 1.x is not guaranteed.

- MMFlow 1.x relies on MMEngine to run. MMEngine is a new foundational library for training deep learning models in OpenMMLab 2.x models. The dependencies of file IO and training are migrated from MMCV 1.x to MMEngine.

- MMFlow 1.x relies on MMCV>=2.0.0rc0. Although MMCV no longer maintains the training functionalities since 2.0.0rc0, MMFlow 1.x relies on the data transforms, CUDA operators, and image processing interfaces in MMCV. Note that the package `mmcv` is the version that provide pre-built CUDA operators and `mmcv-lite` does not since MMCV 2.0.0rc0, while `mmcv-full` has been deprecated.

#### Training and testing

- MMFlow 1.x uses Runner in [MMEngine](https://github.com/open-mmlab/mmengine) rather than that in MMCV. The new Runner implements and unifies the building logic of dataset, model, evaluation, and visualization. Therefore, MMFlow 1.x no longer maintains the building logics of those modules in `mmflow.train.apis` and `tools/train.py`. Those code have been migrated into [MMEngine](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py). Please refer to the [migration guide of Runner in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/runner.html) for more details.

- The Runner in MMEngine also supports testing and validation. The testing scripts are also simplified, which has similar logic as that in training scripts to build the runner.

- The execution points of hooks in the new Runner have been enriched to allow more flexible customization. Please refer to the [migration guide of Hook in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/hook.html) for more details.

- Learning rate and momentum scheduling has been migrated from `Hook` to `Parameter Scheduler` in MMEngine. Please refer to the [migration guide of Parameter Scheduler in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/param_scheduler.html) for more details.

#### Configs

- The [Runner in MMEngine](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py) uses a different config structures to ease the understanding of the components in runner. Users can read the [config example of mmflow](../user_guides/1_config.md) or refer to the [migration guide in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/runner.html) for migration details.

- The file names of configs and models are also refactored to follow the new rules unified across OpenMMLab 2.x projects. Please refer to the [user guides of config](../user_guides/1_config.md) for more details.

#### Dataset

The Dataset classes implemented in MMFlow 1.x all inherits from the [BaseDataset in MMEngine](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html). There are several changes of Dataset in MMFlow 1.x.

- All the datasets support to serialize the data list to reduce the memory when multiple workers are built to accelerate data loading.

- The interfaces are changed accordingly.

#### Data Transforms

The data transforms in MMFlow 1.x all inherits from those in MMCV>=2.0.0rc0, which follows a new convention in OpenMMLab 2.x projects.
The changes are listed as below:

- The interfaces are also changed. Please refer to the [API doc](https://mmflow.readthedocs.io/en/1.x/)
- The functionality of some data transforms (e.g., `Resize`) are decomposed into several transforms.
- The same data transforms in different OpenMMLab 2.x libraries have the same augmentation implementation and the logic of the same arguments, i.e., `Resize` in MMFlow 1.x and MMSeg 1.x will resize the image in the exact same manner given the same arguments.

#### Model

- Model:
- Evaluation
- Visualization

### Improvements

- The training speed of those models with some common training strategies are improved, including those with synchronized batch normalization and mixed precision training.

- Support mixed precision training of all the models. However, some models may got Nan results due to some numerical issues. We will update the documentation and list their results (accuracy of failure) of mixed precision training.

### Ongoing changes

1. Inference interfaces: a unified inference interfaces will be supported in the future to ease the use of released models.

2. Interfaces of useful tools that can be used in notebook: more useful tools that implemented in the `tools` directory will have their python interfaces so that they can be used through notebook and in downstream libraries.

3. Documentation: we will add more design docs, tutorials, and migration guidance so that the community can deep dive into our new design, participate the future development, and smoothly migrate downstream libraries to MMFlow 1.x.

4. SpyNet, Flow1D, CRAFT support.
