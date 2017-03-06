# Implementation of the All Convolution model in keras

## Source

'Striving for Simplicity: The All Convolutional Net' The original paper can be found [here](https://arxiv.org/abs/1412.6806#).

## Requirements

- keras with Tensorflow backend (keras version 1.0.4 or later)
- h5py (if you want to save your model)
- numpy
- pandas (if you want to save the logs of your model)
- cv2 (for image resizing)

## External data

In this implementation we are using the Cifar10 dataset. Either you can import the dataset from keras.datasets

Or

You can Download the dataset from [here](https://www.cs.toronto.edu/~kriz/cifar.html).

## Usage

If you want to run the model without using the pretrained weights then: Run `python allconv.py`

Model and Usage demo: Run `python transfer_learning.py`

You can download the weights from here: [weights.994-0.56.hdf5](https://doc-0s-38-docs.googleusercontent.com/docs/securesc/vpocm17pitg002qmgv62depufpsg2pjb/ll4rh66jflb55d2m3cd08nfd2bo3e6jg/1488794400000/16597602046324852878/16597602046324852878/0B3eKX5eGCnJXWkRubkl2azQ3WXc?e=download)

## Results

The above model easily achieves more than 90% accuracy after the first 350 iterations. If you want to increase the accuracy then you can try much more heavy data augmentation at the cost of computation time.

## Licensing

MIT License
