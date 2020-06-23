# tf-human-action-datasets

## repository organization

- `datasets/`: all stuff related with datasets' loading and manipulating.
- `datasets/common/`: utilities related with joint and labels shared across datasets.
- `datasets/montalbanov2/`: functions and joint definitions from [Montalbano V2](http://chalearnlap.cvc.uab.es/dataset/13/description/) dataset.
- `datasets/montalbanov2/compute_average_lengths.py`: script used to compute average lengths of each skeleton's connection. This measurements are needed in normalization process.
- `datasets/ntu_rbgd/`: script to load and generate `tfrecords` to work with [NTU-RBGD](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp) dataset.
- `datasets/tfrecords/`: set of utilities to encode and decode `tfrecords` data structures.
- `datasets/tfrecords/`: set of utilities to encode and decode `tfrecords` data structures.
- `etc/conf/`: examples of configurations used in some scripts.
- `etc/experiments/gesture_localization/`: folders containing configuration and results of experiments about gesture localization process. They're separated in 3 types: `train_*`, `eval_*` and `search_*`.
- `examples/`: scripts to validate implementations.
- `models/base/`: general stuff used to work with all models.
- `models/gesture_localization/`: scripts used to prepare data, train and validate models to address gesture localization problem.
- `models/options/`: contains a `*.proto` file with all configurations used by the scripts from this repository, as well as a helper function to load and validate configurations.
- `models/skeleton_net/`: scripts created when trying to build a model based on SkeletonNet using NTU-RGBD dataset.
- `tf_patch/`: a monkey patch used to fix a tensorflow [issue](https://github.com/tensorflow/tensorflow/issues/24520), which is probably already fixed.
- `utils/`: scripts and classes used all over this repository.


## references 

### tfrecord

- https://github.com/chiphuyen/stanford-tensorflow-tutorials/blob/master/2017/examples/09_tfrecord_example.py

- https://github.com/tensorflow/tensorflow/blob/v1.12.0/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py

- http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/

- https://medium.com/@dikatok/making-life-better-read-easier-with-tensorflow-dataset-api-fb91851e51f4