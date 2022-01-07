# Changelog

## v0.2.0(07/01/2022)

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
