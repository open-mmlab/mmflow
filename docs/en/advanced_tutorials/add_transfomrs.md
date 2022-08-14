# Adding New Data Transforms

1. Write a new pipeline in any file, e.g., `my_pipeline.py`. It takes a dict as input and return a dict.

    ```python
    from mmflow.datasets import PIPELINES

    @TRANSFORMS.register_module()
    class MyTransform:

        def __call__(self, results):
            results['dummy'] = True
            return results
   ```

2. Import the new class.

    ```python
    from .my_pipeline import MyTransform
    ```

3. Use it in config files.

    ```python
    train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='ColorJitter', brightness=0.5, contrast=0.5, saturation=0.5,
         hue=0.5),
    dict(type='RandomGamma', gamma_range=(0.7, 1.5)),
    dict(type='Normalize', mean=[0., 0., 0.], std=[255., 255., 255.], to_rgb=False),
    dict(type='GaussianNoise', sigma_range=(0, 0.04), clamp_range=(0., 1.)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomAffine',
         global_transform=dict(
            translates=(0.05, 0.05),
            zoom=(1.0, 1.5),
            shear=(0.86, 1.16),
            rotate=(-10., 10.)
        ),
         relative_transform=)dict(
            translates=(0.00375, 0.00375),
            zoom=(0.985, 1.015),
            shear=(1.0, 1.0),
            rotate=(-1.0, 1.0)
        ),
    dict(type='RandomCrop', crop_size=(384, 448)),
    dict(type='MyTransform'),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['imgs', 'flow_gt'],
        meta_keys=('img_fields', 'ann_fields', 'filename1', 'filename2',
                   'ori_filename1', 'ori_filename2', 'filename_flow',
                   'ori_filename_flow', 'ori_shape', 'img_shape',
                   'img_norm_cfg'))]
    ```
