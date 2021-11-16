# optimizer
optimizer = dict(type='Adam', lr=5e-5, weight_decay=0.0004, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='MultiStage',
    by_epoch=False,
    gammas=[0.5, 0.5, 0.5, 0.5, 0.5],
    milestone_lrs=[5e-5, 3e-5, 2e-5, 1e-5, 5e-6],
    milestone_iters=[0, 150000, 300000, 450000, 600000],
    steps=[[
        45000, 65000, 85000, 95000, 97500, 100000, 110000, 120000, 130000,
        140000
    ],
           [
               195000, 215000, 235000, 245000, 247500, 250000, 260000, 270000,
               280000, 290000
           ],
           [
               345000, 365000, 385000, 395000, 397500, 400000, 410000, 420000,
               430000, 440000
           ],
           [
               495000, 515000, 535000, 545000, 547500, 550000, 560000, 570000,
               580000, 590000
           ],
           [
               645000, 665000, 685000, 695000, 697500, 700000, 710000, 720000,
               730000, 740000
           ]])

runner = dict(type='IterBasedRunner', max_iters=750000)
checkpoint_config = dict(by_epoch=False, interval=50000)
evaluation = dict(interval=50000, metric='EPE')
