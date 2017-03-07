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


You can download the weights from here: [weights.994-0.56.hdf5](https://drive.google.com/file/d/0B3eKX5eGCnJXWkRubkl2azQ3WXc/view?usp=sharing)
And use the weights to retrain your model.

## Results

The above model easily achieves more than 90% accuracy after the first 350 iterations. If you want to increase the accuracy then you can try much more heavy data augmentation at the cost of computation time.

## Licensing

MIT License

## Additional Notes

### Use of Scheduler:

In the original paper learning rate of 'γ' and  scheduler S = "e1 ,e2 , e3" were used in which γ is multiplied by a fixed multiplier of 0.1 after e1. e2 and e3 epochs respectively. (where e1 = 200, e2 = 250, e3 = 300)
But in our implmentation we have went with a learning rate of 0.1, decay of 1e-6 and momentum of 0.9. This is done to make the model converge to a desirable accuracy in the first 100 epoch (Benificial for those who have a constrain on the computation power, feel free to play around with the learning rate and scheduler)

### Data Augmentation:

In the original paper very extensive data augmentation were used such as placing the cifar10 images of size 32 × 32 into larger 126 × 126 pixel images and can hence be heavily scaled, rotated and color augmented.
In our implementation we have only done very mild data augmentation. We hope that the accuracy will increase if you increase the data augmentation.
