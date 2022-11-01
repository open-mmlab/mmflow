model = dict(
    type='Flow1D',
    data_preprocessor=dict(
        type='FlowDataPreprocessor',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        bgr_to_rgb=True),
    radius=32,
    cxt_channels=128,
    h_channels=128,
    encoder=dict(
        type='RAFTEncoder',
        in_channels=3,
        out_channels=256,
        net_type='Basic',
        norm_cfg=dict(type='IN'),
        init_cfg=[
            dict(
                type='Kaiming',
                layer=['Conv2d'],
                mode='fan_out',
                nonlinearity='relu'),
            dict(type='Constant', layer=['InstanceNorm2d'], val=1, bias=0)
        ]),
    cxt_encoder=dict(
        type='RAFTEncoder',
        in_channels=3,
        out_channels=256,
        net_type='Basic',
        norm_cfg=dict(type='SyncBN'),
        init_cfg=[
            dict(
                type='Kaiming',
                layer=['Conv2d'],
                mode='fan_out',
                nonlinearity='relu'),
            dict(type='Constant', layer=['SyncBatchNorm2d'], val=1, bias=0)
        ]),
    decoder=dict(
        type='Flow1DDecoder',
        net_type='Basic',
        radius=32,
        iters=24,
        corr_op_cfg=dict(type='CorrLookupFlow1D'),
        gru_type='SeqConv',
        flow_loss=dict(type='SequenceLoss'),
        act_cfg=dict(type='ReLU')),
    freeze_bn=False,
    train_cfg=dict(),
    test_cfg=dict(),
)
