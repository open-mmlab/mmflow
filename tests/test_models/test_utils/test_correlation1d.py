# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor

from mmflow.models.utils.correlation1d import Correlation1D

_feat1 = Tensor(
    [[[[1.0154, 0.4896, 1.8628, 0.0762, 0.2545, -0.1868, 0.5853, 1.6154],
       [-0.4458, -1.3631, -0.6748, 0.2643, 0.8796, 1.2195, -0.9295, 0.3636],
       [0.2345, 0.1408, -0.2794, -2.2829, -1.8497, -0.4348, -0.1259, 1.2991],
       [0.9833, 0.5806, 0.0429, -1.5982, 1.1363, 0.0071, -1.5662, 0.0415]],
      [[-2.5624, 0.4736, 0.3118, -0.1595, 0.4542, -1.2495, -0.3464, -1.1194],
       [0.1017, 1.1922, -1.2911, 0.6752, 1.4180, -0.3162, -0.3809, 1.4444],
       [-0.8802, 1.5789, -0.7804, -0.2817, 0.3465, -0.6741, 0.1570, 0.1059],
       [-0.8849, 0.3025, -0.3609, 0.7738, 0.8476, -0.2813, 1.5131, -1.4178]],
      [[0.2065, -0.8124, -0.6505, 1.6508, 1.7852, 1.2732, 0.4985, -0.5486],
       [2.7083, 1.0688, 0.4090, -0.1851, 1.0733, 1.1038, -1.4032, 0.2552],
       [1.5166, -0.6669, 1.3872, -0.4971, 1.9420, -2.2243, -2.3078, -0.4577],
       [-1.7597, 0.7735, 1.1435, -0.5766, 1.0973, -0.1990, -1.1990, 0.1093]],
      [[0.2446, 1.8493, 1.7110, 1.1204, -1.7352, -1.3811, -0.2492, 0.8741],
       [0.3271, 0.2713, -1.3248, -0.2370, 0.4934, -0.8729, -0.3618, 0.5313],
       [0.8359, -0.2329, 0.4883, 0.1030, 0.2581, 0.3148, -0.9930, 0.2271],
       [-1.1038, 0.0708, -0.4958, -1.1129, -0.9431, -0.0880, 1.0499,
        -0.6881]]]])
_feat2 = Tensor(
    [[[[1.3175, 1.4551, 1.6624, -0.5219, 0.3938, -1.4649, 0.9400, -0.4180],
       [0.4486, 0.0388, -0.6881, -1.4353, 1.8669, 0.6907, 0.0128, 0.2979],
       [1.7176, 0.3644, -1.2154, -1.9436, 0.9357, 2.0734, -0.3146, 0.1123],
       [-0.7050, 1.4828, 0.8406, 0.3374, 0.7549, 0.4404, -0.1620, 0.3539]],
      [[1.1737, -0.9930, -0.6959, -1.7765, -0.4785, -0.5701, -0.6154, 0.8447],
       [2.2322, 1.2820, -0.9384, -0.2065, 0.1662, 0.9703, 0.1947, -0.7589],
       [0.9334, -0.5888, 0.2904, -1.1869, -1.3860, -1.1149, -0.4794, -0.4440],
       [1.0862, -1.1460, 0.9998, -1.3857, 1.0615, -0.1334, 1.4889, -0.2771]],
      [[0.4017, 0.4662, 0.6031, 2.2982, -1.3094, -0.7295, -0.2682, 0.3263],
       [-0.2803, 1.5200, -0.5896, 0.5558, -0.6111, -0.5191, -0.0100, 0.4099],
       [0.3736, -1.0845, -0.9815, 0.9264, 0.5722, -2.2061, 0.9850, -0.2834],
       [0.2425, 1.4829, -0.8054, 1.1259, -1.0513, 1.3195, -1.7388, 0.3673]],
      [[0.0612, 0.3328, 0.1373, -1.9487, 0.8354, -0.7799, -0.4399, 1.7067],
       [1.1250, -0.8651, -0.3540, 0.7884, 1.2341, -1.0060, 1.8890, 0.9911],
       [0.9935, 0.3770, 1.4380, 0.0396, 0.2286, 2.2238, 0.1141, 0.0866],
       [-0.1054, -0.4454, 0.1032, -1.1747, 0.5838, 1.2229, -0.2493, 1.0715]]]])
b, c, h, w = _feat1.size()


def test_correlation():
    gt_corr_x = Tensor([[[[
        -7.8589e-01, 2.0998e+00, 1.8146e+00, 2.0100e+00, 7.7996e-01,
        -1.8402e-01, 1.1842e+00, -1.0520e+00
    ]]],
                        [[[
                            4.9387e-01, 2.3942e-01, 1.2414e-01, -3.2838e+00,
                            1.2874e+00, -9.1842e-01, -2.1343e-01, 1.5433e+00
                        ]]],
                        [[[
                            1.3318e+00, 1.3336e+00, 1.3612e+00, -3.1777e+00,
                            1.4328e+00, -1.8832e+00, 4.9047e-01, 1.0963e+00
                        ]]],
                        [[[
                            3.2244e-01, 7.0587e-01, 6.9355e-01, 9.2706e-01,
                            -5.5962e-01, -1.0494e+00, -3.8291e-01, 1.1421e+00
                        ]]],
                        [[[
                            7.3966e-01, 8.7044e-02, 4.7271e-01, 3.2722e+00,
                            -1.9521e+00, -2.9039e-01, 1.2212e-01, -1.0508e+00
                        ]]],
                        [[[
                            -6.4286e-01, 5.5144e-01, 5.6862e-01, 3.9673e+00,
                            -1.1483e+00, 5.6715e-01, 4.2971e-01, -1.4595e+00
                        ]]],
                        [[[
                            2.7478e-01, 6.7256e-01, 7.4025e-01, 9.7059e-01,
                            -2.3234e-01, -4.1461e-01, 3.6964e-01, -3.9995e-01
                        ]]],
                        [[[
                            3.2379e-01, 1.7486e+00, 1.6268e+00, -9.0931e-01,
                            1.3102e+00, -1.0049e+00, 9.8499e-01, -1.5399e-01
                        ]]],
                        [[[
                            -1.8206e-01, 1.9734e+00, -7.5064e-01, 1.1910e+00,
                            -1.0334e+00, -9.7209e-01, 3.0245e-01, 6.1217e-01
                        ]]],
                        [[[
                            1.0277e+00, 1.4327e+00, -4.5351e-01, 1.2591e+00,
                            -1.3325e+00, -3.0622e-01, 3.5824e-01, -3.0192e-01
                        ]]],
                        [[[
                            -2.3949e+00, 4.3196e-02, 9.5187e-01, 2.0900e-01,
                            -1.6796e+00, -2.9920e-01, -1.3833e+00, -1.8328e-01
                        ]]],
                        [[[
                            7.0550e-01, 3.9977e-01, -3.1122e-01, -4.0425e-01,
                            2.1314e-01, 5.8610e-01, -1.5550e-01, -3.7222e-01
                        ]]],
                        [[[
                            1.9070e+00, 1.5283e+00, -1.3717e+00, -2.8489e-01,
                            9.1540e-01, 4.6496e-01, 6.0432e-01, 5.7434e-02
                        ]]],
                        [[[
                            -7.2508e-01, 1.0374e+00, -4.4210e-01, -8.7988e-01,
                            2.3618e-01, 4.2033e-01, -8.5295e-01, 9.5285e-02
                        ]]],
                        [[[
                            -6.4046e-01, -1.1721e+00, 9.7621e-01, 1.7381e-01,
                            -6.9380e-01, 4.0390e-02, -3.7773e-01, -4.6079e-01
                        ]]],
                        [[[
                            1.9567e+00, 8.9705e-01, -9.7208e-01, -1.2971e-01,
                            7.0929e-01, 4.9284e-01, 6.4348e-01, -1.7833e-01
                        ]]],
                        [[[
                            4.8913e-01, -3.6295e-01, -4.1357e-01, 1.0135e+00,
                            1.2491e+00, -9.6748e-03, 9.6871e-01, 2.9864e-02
                        ]]],
                        [[[
                            6.1752e-01, -1.2145e-01, 3.0352e-01, -1.3873e+00,
                            -1.2457e+00, -2.5753e-01, -7.4235e-01, -2.5819e-01
                        ]]],
                        [[[
                            -1.0247e-01, -4.8132e-01, -2.7320e-01, 1.3869e+00,
                            8.6279e-01, -8.4183e-01, 9.4207e-01, -1.7862e-02
                        ]]],
                        [[[
                            -2.1337e+00, -4.4044e-02, 1.6644e+00, 2.1575e+00,
                            -1.0033e+00, -1.5468e+00, 1.8768e-01, 9.2515e-03
                        ]]],
                        [[[
                            -9.3583e-01, -1.4434e+00, 4.0691e-01, 2.4966e+00,
                            -5.2040e-01, -3.9659e+00, 1.1791e+00, -4.4479e-01
                        ]]],
                        [[[
                            -9.4713e-01, 1.3847e+00, 1.4843e+00, -2.0148e-01,
                            -3.3666e-01, 2.7286e+00, -8.4753e-01, 4.5405e-01
                        ]]],
                        [[[
                            -9.5922e-01, 9.9506e-01, 5.1789e-01, -1.0595e+00,
                            -9.4146e-01, 1.2235e+00, -1.2111e+00, 2.4210e-01
                        ]]],
                        [[[
                            1.1924e+00, 4.9652e-01, -3.8619e-01, -1.5328e+00,
                            4.2940e-01, 2.0451e+00, -4.4219e-01, 1.2412e-01
                        ]]],
                        [[[
                            -9.8240e-01, 1.7715e-01, 6.2259e-01, 4.3668e-01,
                            5.0427e-01, -1.5603e+00, 9.2906e-01, -6.1793e-01
                        ]]],
                        [[[
                            4.9682e-02, 8.1487e-01, 8.7411e-02, 2.8222e-01,
                            -6.2244e-03, 6.6128e-01, -5.0314e-01, 2.4081e-01
                        ]]],
                        [[[
                            -4.6349e-02, 1.1969e+00, -6.4845e-01, 1.1922e+00,
                            -9.2116e-01, 4.8479e-01, -1.2045e+00, 1.9728e-03
                        ]]],
                        [[[
                            9.7235e-01, -1.8080e+00, -1.1013e-01, -4.7668e-01,
                            -2.1431e-01, -1.4644e+00, 1.3455e+00, -1.0921e+00
                        ]]],
                        [[[
                            2.4253e-01, 1.3804e+00, 4.1076e-01, 7.7609e-01,
                            2.6673e-02, 3.4096e-01, -2.9748e-01, -2.2011e-01
                        ]]],
                        [[[
                            -1.7477e-01, 3.8498e-02, -6.2041e-02, 1.3576e-01,
                            -6.7703e-02, -1.6477e-01, -2.6009e-02, -4.3462e-02
                        ]]],
                        [[[
                            1.1731e+00, -3.1510e+00, 6.3514e-01, -2.6042e+00,
                            1.1486e+00, -5.9488e-01, 2.1648e+00, -1.4449e-01
                        ]]],
                        [[[
                            -7.3512e-01, 1.0774e+00, -7.7084e-01, 1.4550e+00,
                            -9.9514e-01, -2.4492e-01, -1.0681e+00, -1.4480e-01
                        ]]]])
    gt_corr_y = Tensor([[[[-0.7859], [-2.5235], [-0.1638], [-1.7374]]],
                        [[[0.2394], [-1.1043], [0.7389], [-0.9226]]],
                        [[[1.3612], [-0.8983], [0.4627], [1.2890]]],
                        [[[0.9271], [0.8622], [0.8074], [0.3946]]],
                        [[[-1.9521], [-1.3409], [0.1167], [-1.1078]]],
                        [[[0.5672], [-0.3065], [-2.4372], [0.0377]]],
                        [[[0.3696], [-0.2678], [0.2223], [-0.7076]]],
                        [[[-0.1540], [0.9861], [0.4548], [0.8085]]],
                        [[[0.3200], [-0.1821], [0.3330], [0.5235]]],
                        [[[-1.2894], [1.4327], [-1.1278], [-0.9617]]],
                        [[[-0.0793], [0.9519], [-0.9306], [-1.1621]]],
                        [[[-0.6505], [-0.4043], [-0.7480], [-0.3882]]],
                        [[[-0.6627], [0.9154], [-0.2077], [0.6645]]],
                        [[[-0.8653], [0.4203], [-0.7476], [0.4841]]],
                        [[[-0.0519], [-0.3777], [-0.4742], [1.0568]]],
                        [[[1.0291], [-0.1783], [-0.3134], [0.1957]]],
                        [[[-0.0319], [-0.6722], [0.4891], [-0.4209]]],
                        [[[-0.8757], [0.6087], [-0.1214], [-1.2429]]],
                        [[[0.4911], [-0.0331], [-0.2732], [-1.0410]]],
                        [[[0.1744], [1.5699], [2.1575], [-0.5303]]],
                        [[[-1.6107], [-2.1319], [-0.5204], [-1.4597]]],
                        [[[1.1992], [-0.0582], [2.7286], [-1.3258]]],
                        [[[0.4204], [-0.9119], [-1.2111], [2.2573]]],
                        [[[-0.1077], [0.1721], [0.1241], [0.2528]]],
                        [[[-0.2588], [-1.1413], [-0.4455], [-0.9824]]],
                        [[[0.4643], [0.7624], [-0.3894], [0.8149]]],
                        [[[0.4720], [-0.0948], [-0.9961], [-0.6485]]],
                        [[[0.1515], [0.4681], [0.8048], [-0.4767]]],
                        [[[-1.0914], [0.2139], [0.1504], [0.0267]]],
                        [[[0.1819], [-0.0381], [0.2858], [-0.1648]]],
                        [[[-1.2718], [1.1349], [-0.6469], [2.1648]]],
                        [[[-1.1768], [0.2256], [0.2718], [-0.1448]]]])
    corr = Correlation1D()
    correlation = corr(_feat1, _feat2, _feat2)
    assert correlation[0].size() == (b * h * w, 1, 1, w)
    assert correlation[1].size() == (b * h * w, 1, h, 1)
    assert torch.allclose(correlation[0], gt_corr_x, atol=1e-4)
    assert torch.allclose(correlation[1], gt_corr_y, atol=1e-4)
