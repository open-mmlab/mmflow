# Changelog

## v0.5.1(07/29/2022)

### Improvements

- Set the maximum version of MMCV to 1.7.0 ([167](https://github.com/open-mmlab/mmflow/pull/167))
- Update the qq_group_qrcode image in resources ([166](https://github.com/open-mmlab/mmflow/pull/166))

### New Contributors

- @Weepingchestnut made their first contribution in https://github.com/open-mmlab/mmflow/pull/166

## v0.5.0(07/01/2022)

### Highlight

- Add config and pre-trained model for FlowNet2 on FlyingChairs ([163](https://github.com/open-mmlab/mmflow/pull/163))

### Documentation

- Add a template for PR ([160](https://github.com/open-mmlab/mmflow/pull/160))
- Fix config file error in metafile ([151](https://github.com/open-mmlab/mmflow/pull/151))
- Fix broken URL in metafile ([157](https://github.com/open-mmlab/mmflow/pull/157))
- Fix broken URLs for issue reporting in README ([147](https://github.com/open-mmlab/mmflow/pull/147))

### Improvements

- Add mim to extras_require in setup.py ([154](https://github.com/open-mmlab/mmflow/pull/154))
- Fix mdformat version to support python3.6 and remove ruby install ([153](https://github.com/open-mmlab/mmflow/pull/153))
- Add test_mim.yml for testing commands of mim in CI ([158](https://github.com/open-mmlab/mmflow/pull/158))

## v0.4.2(05/31/2022)

### Bug Fixes

- Inference bug for sparse flow map ([133](https://github.com/open-mmlab/mmflow/pull/133))
- H and W input images must be divisible by 2\*\*6 ([136](https://github.com/open-mmlab/mmflow/pull/136))

### Documentation

- Configure Myst-parser to parse anchor tag ([129](https://github.com/open-mmlab/mmflow/pull/129))
- Replace markdownlint with mdformat for avoiding installing ruby ([130](https://github.com/open-mmlab/mmflow/pull/130))
- Rewrite install and README by ([139](https://github.com/open-mmlab/mmflow/pull/139), [140](https://github.com/open-mmlab/mmflow/pull/140),
  [141](https://github.com/open-mmlab/mmflow/pull/141), [144](https://github.com/open-mmlab/mmflow/pull/144), [145](https://github.com/open-mmlab/mmflow/pull/145))

## v0.4.1(04/29/2022)

### Feature

- Loading flow annotation from file client ([#116](https://github.com/open-mmlab/mmflow/pull/116))
- Support overall dastaloader settings ([#117](https://github.com/open-mmlab/mmflow/pull/117))
- Generate ann_file for flyingchairs ([121](https://github.com/open-mmlab/mmflow/pull/121))

### Improvements

- Add GPG keys in CI([127](https://github.com/open-mmlab/mmflow/pull/127))

### Bug Fixes

- The config and weights are not corresponding in the metafile.yml ([#118](https://github.com/open-mmlab/mmflow/pull/118))
- Replace recommonmark with myst_parser ([#120](https://github.com/open-mmlab/mmflow/pull/120))

### Documentation

- Add zh-cn doc 0_config\_.md ([#126](https://github.com/open-mmlab/mmflow/pull/126))

## New Contributors

- @HiiiXinyiii made their first contribution in https://github.com/open-mmlab/mmflow/pull/118
- @SheffieldCao made their first contribution in https://github.com/open-mmlab/mmflow/pull/126

## v0.4.0(04/01/2022)

### Highlights

- Support occlusion estimation methods including flow forward-backward consistency, range map of the backward flow, and flow forward-backward abstract difference

### Features

- Support three occlusion estimation methods (#106)
- Support different seeds on different ranks when distributed training (#104)

### Improvements

- Revise collect_env for win platform (#112)
- Add script and documentation for multi-machine distributed training (#107)

## v0.3.0(03/04/2022)

### Highlights

- Officially support CPU Train/Inference
- Add census loss, SSIM loss and smoothness loss
- Officially support model inference in windows platform
- Update `nan` files in Flyingthings3d_subset dataset

### Features

- Add census loss (#100)
- Add smoothness loss function (#97)
- Add SSIM loss function (#96)

### Bug Fixes

- Update `nan` files in Flyingthings3d_subset (#94)
- Add pretrained pwcnet-model when train PWCNet+ (#99)
- Fix bug in non-distributed multi-gpu training/testing (#85)
- Fix writing flow map bug in test (#83)

### Improvements

- Add win-ci (#92)
- Update the installation of MMCV (#89)
- Upgrade isort in pre-commit hook (#87)
- Support CPU Train/Inference (#86)
- Add multi-processes script (#79)
- Deprecate the support for "python setup.py test" (#73)

### Documentation

- Fix broken URLs in GMA README (#93)
- Fix date format in readme (#90)
- Reorganizing OpenMMLab projects in readme (#98)
- Fix README files of algorithms (#84)
- Add url of OpenMMLab and platform in README (76)

## v0.2.0(01/07/2022)

### Highlights

- Support [GMA](../../configs/gma/README.md): Learning to Estimate Hidden Motions with Global Motion Aggregation (ICCV 2021) (#32)
- Fix the bug of wrong refine iter in RAFT, and update [RAFT](../../configs/raft/README.md) model checkpoint after the bug fixing (#62, #68)
- Support resuming from the latest checkpoint automatically (#71)

### Features

- Add `scale_as_level` for multi-level flow loss (#58)
- Add `scale_mode` for correlation block (#56)
- Add `upsample_cfg` in IRR-PWC decoder (#53)

### Bug Fixes

- Resized input image must be dividable by 2^6 (#65)
- Fix RAFT wrong refine iter after evaluation (#62)

### Improvements

- Add `persistent_workers=True` in `val_dataloader` (#63)
- Revise `env_info` key (#46)
- Add digital version (#43)
- Try to create a symbolic link on windows (#37)
- Set a random seed when the user does not set a seed (#27)

### Refactors

- Refactor utils in models (#50)

### Documentation

- Refactor documentation (#14)
- Fix script bug in FlyingChairs dataset prepare (#21)
- Fix broken links in model_zoo (#60)
- Update metafile (#39, #41, #49)
- Update documentation (#28, #35, #36, #47, #48, #70)
