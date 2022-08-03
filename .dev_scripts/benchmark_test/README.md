# Tutorial: Benchmark Test

MMFlow has been upgraded from version 1.0 to version 2.0, with a more extensible framework and a more concise way of use.
At the same time, the model we pre-trained under 1.0 can still be successfully loaded and fine-tuned in 2.0.
We use benchmark test to verify the accuracy metrics of the pre-trained models in 2.0.
Under the right circumstances, these metrics should align with their counterparts in 1.0.
This tutorial will show how to perform benchmark test, including:

- Download all pre-trained models in 1.0.
- Test the accuracy of the pre-trained models.

## Download pre-trained models

For every model in the [model zoo](../../docs/en/model_zoo.md), the download urls for their pre-trained model can be obtained in README.md of their respective configuration folder.
For example, the pre-trained model download url for flownet can be found [here](../../configs/flownet/README.md).
To download all pre-trained models, we just need to get their download urls from the README.md in each configuration folder and download them.

Simply run:
```shell
python .dev_scripts/benchmark_test/download_models.py
```
Then all pre-trained models will be automatically downloaded in `work_dirs/download/hub/checkpoints`.
We recommend NOT changing this default download path, although it doesn't look particularly neat.
Otherwise, subsequent tests may be affected.

## Test pre-trained models

As we know, MMFlow uses config file to train, test or infer models.
The `metafile.yml` in each configuration folder contains the download url of the pre-trained model and configuration file that trained the model.
In the benchmark test, we can utilize this connection between the pre-trained model and config to achieve convenient tests.

Test on single GPU, simply run:
```shell
python .dev_scripts/benchmark_test/test_models.py
```
This command will test the models one by one and automatically generate a folder under `work_dirs` to save the running records and final results.

Manage jobs with slurm, when testing on a Slurm partition named `dev`, simply run:
```shell
python .dev_scripts/benchmark_test/test_models.py --partition dev --use-slurm
```
This command will start multiple threads to test different models in parallel, each thread tests one model.
Up to 8 threads can be run at the same time.
You can change the upper limit of thread by adjusting the number in `sem = threading.Semaphore(8)` in `test_models.py`.
