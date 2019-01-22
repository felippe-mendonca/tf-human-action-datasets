Gesture Localization
===

This is a MLP (Multi-Layer Perceptron) model used to localize whether a gesture is being executed or not. To do so, a mocap (motion capture) stream is transformed into a set a features based on [[1](#references),[2](#references),[3](#references)]. To develop and evauluate this model, a famous dataset of gesture recognition were used: [MontalbanoV2](http://chalearnlap.cvc.uab.es/dataset/13/description/).

Before train and infer your model, you must need to prepare your dataset. First of all, download the train and validation parts on the link given previously. Only the csv files containing labels and skeletons are needed. Put the `SampleXXXX_*.csv` files of each part (train and validation) on a folder with its name.

After that, you'll need to compute the average distances of some body limbs. Do this executing the module `datasets.montalbanov2.compute_average_lengths`. For more information read de help message of this script. Finally, for the training process it'll be necessary to create `*.tfrecords` files. Do that running the module located at this folder: `models.gesture_localization.create_tfrecords`.

## References

1. **Multi-scale deep learning for gesture detection and localization.** *Neverova, N., Wolf, C., Taylor, G. W., & Nebout, F.* (2014). [link](https://nneverova.github.io/npapers/eccvw2014.pdf)

2. **ModDrop: Adaptive multi-modal gesture recognition.** *Neverova, N., Wolf, C., Taylor, G., & Nebout, F.* (2016). [link](https://arxiv.org/abs/1501.00102)

3. **The Moving Pose: An Efficient 3D Kinematics Descriptor for Low-Latency Action Recognition and Detection.** *Mihai Zanfir, Marius Leordeanu, Cristian Sminchisescu* (2013) [link](http://openaccess.thecvf.com/content_iccv_2013/papers/Zanfir_The_Moving_Pose_2013_ICCV_paper.pdf)