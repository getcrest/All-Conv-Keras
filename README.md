# Implementation of the All Convolution model in keras

## Source

Original paper can be found [here]((https://arxiv.org/abs/1412.6806#)).

## Requirements

- keras with Tensorflow backend (keras version 1.0.4 or later)
- h5py (if you want to save your model)
- numpy
- pandas (if you want to save the logs of your model)

## External data

In this implementation we are using the Cifar10 dataset. Either you can import the dataset from keras.datasets

Or

You can Download the dataset from [here](https://www.cs.toronto.edu/~kriz/cifar.html).

## Results

The above model easily achieves more than 90% accuracy after the first 350 iterations. If you want to increase the accuracy then you can try much more heavy data augmentation at the cost of computation time.

## Licensing

MIT License
