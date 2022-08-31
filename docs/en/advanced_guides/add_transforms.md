# Adding New Data Transforms

1. Write a new pipeline in any file, e.g., `my_transform.py`. It takes a dict as input and return a dict.

   ```python
   from mmflow.registry import TRANSFORMS

   @TRANSFORMS.register_module()
   class MyTransform:

       def transforms(self, results):
           results['dummy'] = True
           return results
   ```

2. Import the new class.

   ```python
   from .my_transform import MyTransform
   ```

3. Use it in config files.

   ```python
   train_pipeline = [
   dict(type='LoadImageFromFile'),
   dict(type='LoadAnnotations'),
   dict(type='ColorJitter', brightness=0.5, contrast=0.5, saturation=0.5,
        hue=0.5),
   dict(type='RandomGamma', gamma_range=(0.7, 1.5)),
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
   dict(type='PackFlowInputs')]
   ```
