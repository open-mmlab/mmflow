# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import numpy as np

from .base_dataset import BaseDataset
from .builder import DATASETS

DATASET_SIZE = 22872
VALIDATE_INDICES = [
    5, 17, 42, 45, 58, 62, 96, 111, 117, 120, 121, 131, 132, 152, 160, 248,
    263, 264, 291, 293, 295, 299, 316, 320, 336, 337, 343, 358, 399, 401, 429,
    438, 468, 476, 494, 509, 528, 531, 572, 581, 583, 588, 593, 681, 688, 696,
    714, 767, 786, 810, 825, 836, 841, 883, 917, 937, 942, 970, 974, 980, 1016,
    1043, 1064, 1118, 1121, 1133, 1153, 1155, 1158, 1159, 1173, 1187, 1219,
    1237, 1238, 1259, 1266, 1278, 1296, 1354, 1378, 1387, 1494, 1508, 1518,
    1574, 1601, 1614, 1668, 1673, 1699, 1712, 1714, 1737, 1841, 1872, 1879,
    1901, 1921, 1934, 1961, 1967, 1978, 2018, 2030, 2039, 2043, 2061, 2113,
    2204, 2216, 2236, 2250, 2274, 2292, 2310, 2342, 2359, 2374, 2382, 2399,
    2415, 2419, 2483, 2502, 2504, 2576, 2589, 2590, 2622, 2624, 2636, 2651,
    2655, 2658, 2659, 2664, 2672, 2706, 2707, 2709, 2725, 2732, 2761, 2827,
    2864, 2866, 2905, 2922, 2929, 2966, 2972, 2993, 3010, 3025, 3031, 3040,
    3041, 3070, 3113, 3124, 3129, 3137, 3141, 3157, 3183, 3206, 3219, 3247,
    3253, 3272, 3276, 3321, 3328, 3333, 3338, 3341, 3346, 3351, 3396, 3419,
    3430, 3433, 3448, 3455, 3463, 3503, 3526, 3529, 3537, 3555, 3577, 3584,
    3591, 3594, 3597, 3603, 3613, 3615, 3670, 3676, 3678, 3697, 3723, 3728,
    3734, 3745, 3750, 3752, 3779, 3782, 3813, 3817, 3819, 3854, 3885, 3944,
    3947, 3970, 3985, 4011, 4022, 4071, 4075, 4132, 4158, 4167, 4190, 4194,
    4207, 4246, 4249, 4298, 4307, 4317, 4318, 4319, 4320, 4382, 4399, 4401,
    4407, 4416, 4423, 4484, 4491, 4493, 4517, 4525, 4538, 4578, 4606, 4609,
    4620, 4623, 4637, 4646, 4662, 4668, 4716, 4739, 4747, 4770, 4774, 4776,
    4785, 4800, 4845, 4863, 4891, 4904, 4922, 4925, 4956, 4963, 4964, 4994,
    5011, 5019, 5036, 5038, 5041, 5055, 5118, 5122, 5130, 5162, 5164, 5178,
    5196, 5227, 5266, 5270, 5273, 5279, 5299, 5310, 5314, 5363, 5375, 5384,
    5393, 5414, 5417, 5433, 5448, 5494, 5505, 5509, 5525, 5566, 5581, 5602,
    5609, 5620, 5653, 5670, 5678, 5690, 5700, 5703, 5724, 5752, 5765, 5803,
    5811, 5860, 5881, 5895, 5912, 5915, 5940, 5952, 5966, 5977, 5988, 6007,
    6037, 6061, 6069, 6080, 6111, 6127, 6146, 6161, 6166, 6168, 6178, 6182,
    6190, 6220, 6235, 6253, 6270, 6343, 6372, 6379, 6410, 6411, 6442, 6453,
    6481, 6498, 6500, 6509, 6532, 6541, 6543, 6560, 6576, 6580, 6594, 6595,
    6609, 6625, 6629, 6644, 6658, 6673, 6680, 6698, 6699, 6702, 6705, 6741,
    6759, 6785, 6792, 6794, 6809, 6810, 6830, 6838, 6869, 6871, 6889, 6925,
    6995, 7003, 7026, 7029, 7080, 7082, 7097, 7102, 7116, 7165, 7200, 7232,
    7271, 7282, 7324, 7333, 7335, 7372, 7387, 7407, 7472, 7474, 7482, 7489,
    7499, 7516, 7533, 7536, 7566, 7620, 7654, 7691, 7704, 7722, 7746, 7750,
    7773, 7806, 7821, 7827, 7851, 7873, 7880, 7884, 7904, 7912, 7948, 7964,
    7965, 7984, 7989, 7992, 8035, 8050, 8074, 8091, 8094, 8113, 8116, 8151,
    8159, 8171, 8179, 8194, 8195, 8239, 8263, 8290, 8295, 8312, 8367, 8374,
    8387, 8407, 8437, 8439, 8518, 8556, 8588, 8597, 8601, 8651, 8657, 8723,
    8759, 8763, 8785, 8802, 8813, 8826, 8854, 8856, 8866, 8918, 8922, 8923,
    8932, 8958, 8967, 9003, 9018, 9078, 9095, 9104, 9112, 9129, 9147, 9170,
    9171, 9197, 9200, 9249, 9253, 9270, 9282, 9288, 9295, 9321, 9323, 9324,
    9347, 9399, 9403, 9417, 9426, 9427, 9439, 9468, 9486, 9496, 9511, 9516,
    9518, 9529, 9557, 9563, 9564, 9584, 9586, 9591, 9599, 9600, 9601, 9632,
    9654, 9667, 9678, 9696, 9716, 9723, 9740, 9820, 9824, 9825, 9828, 9863,
    9866, 9868, 9889, 9929, 9938, 9953, 9967, 10019, 10020, 10025, 10059,
    10111, 10118, 10125, 10174, 10194, 10201, 10202, 10220, 10221, 10226,
    10242, 10250, 10276, 10295, 10302, 10305, 10327, 10351, 10360, 10369,
    10393, 10407, 10438, 10455, 10463, 10465, 10470, 10478, 10503, 10508,
    10509, 10809, 11080, 11331, 11607, 11610, 11864, 12390, 12393, 12396,
    12399, 12671, 12921, 12930, 13178, 13453, 13717, 14499, 14517, 14775,
    15297, 15556, 15834, 15839, 16126, 16127, 16386, 16633, 16644, 16651,
    17166, 17169, 17958, 17959, 17962, 18224, 21176, 21180, 21190, 21802,
    21803, 21806, 22584, 22857, 22858, 22866
]


@DATASETS.register_module()
class FlyingChairsOcc(BaseDataset):
    """FlyingChairsOcc dataset."""

    def __init__(self, *args, **kwargs):

        self.split = np.ones(DATASET_SIZE)
        self.split[VALIDATE_INDICES] = 2

        super().__init__(*args, **kwargs)

    def load_img_info(self, img1_filename, img2_filename):
        """Load information of image1 and image2.

        Args:
            img1_filename (list): ordered list of abstract file path of img1.
            img2_filename (list): ordered list of abstract file path of img2.
        """

        num_file = len(img1_filename)
        for i in range(num_file):
            if (not self.test_mode
                    and self.split[i] == 1) or (self.test_mode
                                                and self.split[i] == 2):
                data_info = dict(
                    img_info=dict(
                        filename1=img1_filename[i],
                        filename2=img2_filename[i]),
                    ann_info=dict())
                self.data_infos.append(data_info)

    def load_ann_info(self, filename, filename_key):
        """Load information of optical flow.

        This function splits the dataset into two subsets, training subset and
        testing subset.

        Args:
            filename (list): ordered list of abstract file path of annotation.
            filename_key (str): the annotation key for FlyingChairsOcc dataset
                'flow_fw', 'flow_bw', 'occ_fw', and 'occ_bw'.
        """

        num_files = len(filename)
        count = 0
        for i in range(num_files):
            if (not self.test_mode and self.split[i] == 1) \
                    or (self.test_mode and self.split[i] == 2):
                self.data_infos[count]['ann_info'][filename_key] = filename[i]
                count += 1

    def load_data_info(self):
        """Load data information, including file path of image1, image2 and
        optical flow."""

        # unpack FlyingChairsOcc directly, will see `data` subdirctory.
        self.img1_dir = osp.join(self.data_root, 'data')
        self.img2_dir = osp.join(self.data_root, 'data')
        self.flow_dir = osp.join(self.data_root, 'data')
        self.occ_dir = osp.join(self.data_root, 'data')

        # data in FlyingChairsOcc dataset has specific suffix
        self.img1_suffix = '_img1.png'
        self.img2_suffix = '_img2.png'
        self.flow_fw_suffix = '_flow.flo'
        self.flow_bw_suffix = '_flow_b.flo'
        self.occ_fw_suffix = '_occ1.png'
        self.occ_bw_suffix = '_occ2.png'

        img1_filenames = self.get_data_filename(self.img1_dir,
                                                self.img1_suffix)
        img2_filenames = self.get_data_filename(self.img2_dir,
                                                self.img2_suffix)
        flow_fw_filenames = self.get_data_filename(self.flow_dir,
                                                   self.flow_fw_suffix)
        flow_bw_filenames = self.get_data_filename(self.flow_dir,
                                                   self.flow_bw_suffix)
        occ_fw_filenames = self.get_data_filename(self.occ_dir,
                                                  self.occ_fw_suffix)
        occ_bw_filenames = self.get_data_filename(self.occ_dir,
                                                  self.occ_bw_suffix)

        assert len(img1_filenames) == len(img2_filenames) == len(
            flow_fw_filenames) == len(flow_bw_filenames) == len(
                occ_fw_filenames) == len(occ_bw_filenames)

        self.load_img_info(img1_filenames, img2_filenames)
        self.load_ann_info(flow_fw_filenames, 'filename_flow_fw')
        self.load_ann_info(flow_bw_filenames, 'filename_flow_bw')
        self.load_ann_info(occ_fw_filenames, 'filename_occ_fw')
        self.load_ann_info(occ_bw_filenames, 'filename_occ_bw')
