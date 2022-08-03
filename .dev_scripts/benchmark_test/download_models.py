# Copyright (c) OpenMMLab. All rights reserved.

# This tool is used to download all models in configs.

import glob
import os
import os.path as osp
import platform
import re
import sys

MMFlow_ROOT = osp.dirname(osp.dirname(osp.dirname(__file__)))
DOWNLOAD_DIR = osp.join(MMFlow_ROOT, 'work_dirs', 'download')
IS_WINDOWS = (platform.system() == 'Windows')


def find_pth_files(file: str):
    """Find all strs of pth files from a file.

    Args:
        file (str): The original file (config or README).
    Returns:
        List[str]: List of pth files.
    """

    with open(file, 'r', encoding='utf-8') as f:
        data = f.read()

    if file.endswith('md'):
        pth_files = re.findall(r'=.?(https://.*?\.pth)', data)
    else:
        pth_files = re.findall(r'=.?\'(https?://.*?\.pth)\'', data, re.S)

    return (pth_files)


def find_all_pth(md_file):
    """Find all pre-trained checkpoints of a method (pth).

    Args:
        md_file (str): Path to .md file.
    Returns:
        Bool: If the target .pth files are downloaded successfully.
    """

    md_file = md_file.replace(os.sep, '/')
    config_dir, _ = osp.split(md_file)
    files = os.listdir(config_dir)
    config_files = [
        osp.join(config_dir, file) for file in files if file.endswith('.py')
    ]
    all_files = config_files + [md_file]
    pth_files = []
    for file in all_files:
        sub_list = find_pth_files(file)
        if len(sub_list) > 0:
            pth_files.extend(sub_list)
    return pth_files


def download_pth(pth_files):
    """Download all pth files.

    Args:
        pth_files (List[str]): List of pth files.
    """

    # clear
    def clear_path(path: str):
        path = path.replace(' ', '')
        path = path.replace('\'', '')
        path = path.replace('\\', '')
        path = path.replace('+', '')
        path = path.replace('\n', '')
        return path

    pth_files = [clear_path(file) for file in pth_files]
    pth_files = list(set(pth_files))
    pth_files.sort()

    checkpoints_path = osp.join(DOWNLOAD_DIR, 'hub', 'checkpoints')
    if not osp.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    for url in pth_files:
        path = osp.join(checkpoints_path, osp.basename(url))
        print(f'download {path} from {url}')
        if IS_WINDOWS:
            import wget
            wget.download(url, path)
        else:
            os.system(f'wget -N -P {checkpoints_path} {url}')


if __name__ == '__main__':
    configs_root = osp.join(MMFlow_ROOT, 'configs')
    file_list = glob.glob(
        osp.join(configs_root, '**', '*README.md'), recursive=True)
    file_list.sort()

    if not file_list:
        sys.exit(0)

    pth_files = []
    for fn in file_list:
        pth_files.extend(find_all_pth(fn))

    download_pth(pth_files)
