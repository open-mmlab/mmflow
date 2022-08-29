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
