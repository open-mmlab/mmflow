# Copyright (c) OpenMMLab. All rights reserved.

# This tool is used for benchmark-test.

import datetime
import glob
import os
import os.path as osp
import platform
import sys
import threading

import yaml

MMFlow_ROOT = osp.dirname(osp.dirname(osp.dirname(__file__)))
DOWNLOAD_DIR = osp.join(MMFlow_ROOT, 'work_dirs', 'download')
LOG_DIR = osp.join(
    MMFlow_ROOT, 'work_dirs',
    'benchmark_test_' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
PARTITION = 'mm_seg'
IS_WINDOWS = (platform.system() == 'Windows')

MODEL_TYPES = {
    'flownet': ['flownetc', 'flownets'],
    'flownet2':
    ['flownet2', 'flownet2cs', 'flownet2css', 'flownet2sd', 'flownet2css-sd'],
    'pwcnet': ['pwcnet'],
    'liteflownet': ['liteflownet'],
    'liteflownet2': ['liteflownet2'],
    'irr': ['irrpwc'],
    'maskflownet': ['maskflownet', 'maskflownets'],
    'gma': ['gma'],
    'raft': ['raft']
}

sem = threading.Semaphore(8)  # The maximum number of restricted threads


def find_valid_checkpoints(model_type):
    """Find valid checkpoints in checkpoint dir according to model_type.

    For example, if `model_type` is `flownet`, all flownet related pth file,
        such as `flownetc_8x1_sfine_sintel_384x448.pth` will be found. This
        function is designed for storage constraints.
    Args:
        model_type (str): The type of model, such as `flownet`.
    Returns:
        List(str): Checkpoints that are related to `model_type`.
    """
    checkpoints = os.listdir(osp.join(DOWNLOAD_DIR, 'hub', 'checkpoints'))
    valid_checkpoints = []
    for ckpt in checkpoints:
        ckpt_prefix = ckpt.split('_')[0]
        for model_prefix in MODEL_TYPES[model_type]:
            if model_prefix == ckpt_prefix:
                valid_checkpoints.append(ckpt)
                break
    return valid_checkpoints


def find_available_port():
    """Find an available port."""

    port = 65535
    while True:
        if IS_WINDOWS:
            port_inuse = os.popen('netstat -an | findstr :' +
                                  str(port)).readlines()
        else:
            port_inuse = os.popen('netstat -antu | grep :' +
                                  str(port)).readlines()
        if not port_inuse:
            yield port
        port -= 1
        if port < 1024:
            port = 65535


def slurm_test(info: dict, thread_id: int, allotted_port: int):
    """Slurm test.

    Args:
        info (dict): Test info from metafile.yml.
        thread_id (int): The id of thread.
        allotted_port (int): The id of allotted port.
    """

    sem.acquire()

    config = info['Config']
    weights = info['Weights']

    config = config.replace(' ', '')
    weights = weights.replace(' ', '')
    basename, _ = osp.splitext(osp.basename(config))
    weights = osp.join(DOWNLOAD_DIR, 'hub', 'checkpoints',
                       osp.basename(weights))
    assert osp.exists(weights), f'file {weights} doesn\'t exist'

    env_cmd = 'SRUN_ARGS="--quotatype=spot" '
    env_cmd += f'TORCH_HOME={DOWNLOAD_DIR} MASTER_PORT={allotted_port} '
    env_cmd += 'GPUS=1 GPUS_PER_NODE=1'
    base_cmd = 'sh tools/slurm_test.sh'
    task_cmd = f'{PARTITION} {basename}'
    out_file = osp.join(LOG_DIR, f'{thread_id:03d}_{basename}.log')
    cmd = f'{env_cmd} {base_cmd} {task_cmd} {config} {weights}' \
          f' > {out_file} 2>&1'

    print(f'RUN {thread_id:03d}: {cmd}')
    os.system(cmd)

    sem.release()


def py_test(info: dict):
    """Test on single-gpu.

    Args:
        info (dict): Test info from metafile.yml.
    """

    config = info['Config']
    weights = info['Weights']

    config = config.replace(' ', '')
    weights = weights.replace(' ', '')
    basename, _ = osp.splitext(osp.basename(config))
    weights = osp.join(DOWNLOAD_DIR, 'hub', 'checkpoints',
                       osp.basename(weights))
    assert osp.exists(weights), f'file {weights} doesn\'t exist'

    out_file = osp.join(LOG_DIR, f'{basename}.log')
    cmd = f'TORCH_HOME={DOWNLOAD_DIR} python tools/test.py {config} ' \
          f'{weights} > {out_file} 2>&1'
    print(f'RUN {cmd}')
    os.system(cmd)


def test_models(meta_file: str, slurm_style: bool, available_ports):
    """Test all models in a metafile.

    Args:
        meta_file (str): The path of metafile.yml.
        slurm_style (bool): Use slurm_test.sh if `slurm_style=True`.
        available_ports: Available ports.
    """

    global thread_num

    _, model_type = osp.split(osp.split(meta_file)[0])
    assert model_type in MODEL_TYPES, f'the model type ({model_type}) ' \
                                      f'of metafile.yml cannot be recognized'

    valid_checkpoints = find_valid_checkpoints(model_type)

    if len(valid_checkpoints) > 0:
        with open(meta_file, 'r', encoding='utf-8') as f:
            data = f.read()
        yaml_data = yaml.load(data, yaml.FullLoader)

        for i in range(len(yaml_data['Models'])):
            info = yaml_data['Models'][i]
            if slurm_style:
                allotted_port = next(available_ports)
                threading.Thread(
                    target=slurm_test,
                    args=(info, thread_num, allotted_port)).start()
                thread_num += 1
            else:
                py_test(info)
    else:
        download_path = osp.join(DOWNLOAD_DIR, 'hub', 'checkpoints')
        print(f'The checkpoints of {model_type} are not '
              f'downloaded in {download_path}')


if __name__ == '__main__':
    configs_root = osp.join(MMFlow_ROOT, 'configs')
    file_list = glob.glob(
        osp.join(configs_root, '**', '*metafile.yml'), recursive=True)
    file_list.sort()

    if not file_list:
        sys.exit(0)

    if not osp.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    thread_num = 0
    available_ports = find_available_port()
    for fn in file_list:
        test_models(fn, True, available_ports)
