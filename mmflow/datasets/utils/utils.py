# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Optional, Sequence, Union

from mmengine.utils import scandir


def get_data_filename(
        data_dirs: Union[Sequence[str], str],
        suffix: Optional[str] = None,
        exclude: Optional[Sequence[str]] = None) -> Sequence[str]:
    """Get file name from data directory.

    Args:
        data_dirs (list, str): the directory of data
        suffix (str, optional): the suffix for data file. Defaults to None.
        exclude (list, optional): list of files will be excluded.

    Returns:
        list: the list of data file.
    """

    if data_dirs is None:
        return None
    data_dirs = data_dirs \
        if isinstance(data_dirs, (list, tuple)) else [data_dirs]

    suffix = '' if suffix is None else suffix

    if exclude is None:
        exclude = []
    else:
        assert isinstance(exclude, (list, tuple))

    files = []
    for data_dir in data_dirs:
        for f in scandir(data_dir, suffix=suffix):
            if f not in exclude:
                files.append(osp.join(data_dir, f))
    files.sort()
    return files


def load_img_info(data_infos: Sequence[dict], img1_filename: Sequence[str],
                  img2_filename: Sequence[str]) -> None:
    """Load information of images.

    Args:
        data_infos (list): data information.
        img1_filename (list): ordered list of abstract file path of img1.
        img2_filename (list): ordered list of abstract file path of img2.
    """

    num_file = len(img1_filename)
    for i in range(num_file):
        data_info = dict(
            img1_path=img1_filename[i], img2_path=img2_filename[i])
        data_infos.append(data_info)


def load_ann_info(data_infos: Sequence[dict], filename: Sequence[str],
                  filename_key: str) -> None:
    """Load information of annotation.

    Args:
        data_infos (list): data information.
        filename (list): ordered list of abstract file path of annotation.
        filename_key (str): the annotation key e.g. 'flow'.
    """
    assert len(filename) == len(data_infos)
    num_files = len(filename)
    for i in range(num_files):
        data_infos[i][filename_key] = filename[i]
