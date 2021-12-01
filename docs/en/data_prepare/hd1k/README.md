# Prepare hd1k dataset

<!-- [DATASET] -->

```bibtex
@inproceedings{kondermann2016hci,
  title={The HCI Benchmark Suite: Stereo and Flow Ground Truth With Uncertainties for Urban Autonomous Driving},
  author={Kondermann, Daniel and Nair, Rahul and Honauer, Katrin and Krispin, Karsten and Andrulis, Jonas and Brock, Alexander and Gussefeld, Burkhard and Rahimimoghaddam, Mohsen and Hofmann, Sabine and Brenner, Claus and others},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  pages={19--28},
  year={2016}
}
```

```text
hd1k
|   |   ├── hd1k_flow_gt
|   |   |    ├── flow_occ
|   |   |    |     ├── xxxxxx_xxxx.png
|   |   ├── hd1k_input
|   |   |    ├── image_2
|   |   |    |     ├── xxxxxx_xxxx.png
```

You can download datasets on this [webpage](http://hci-benchmark.iwr.uni-heidelberg.de/). Then, you need to unzip and move corresponding datasets to follow the folder structure shown above. The datasets have been well-prepared by the original authors.
