# Changelog

# v0.4.0(04/01/2022)

### Highlights

- Support occlusion estimation methods including flow forward-backward consistency, range map of the backward flow, and flow forward-backward abstract difference

### Features

- Support three occlusion estimation methods (#106)
- Support different seeds to different ranks when distributed training (#104)

### Improvements

- Revise collect_env for win platform (112)
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

### Documents

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

### Documents

- Refactor documentation (#14)
- Fix script bug in FlyingChairs dataset prepare (#21)
- Fix broken links in model_zoo (#60)
- Update metafile (#39, #41, #49)
- Update documentation (#28, #35, #36, #47, #48, #70)
