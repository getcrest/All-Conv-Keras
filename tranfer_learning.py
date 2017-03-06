from keras.models import Sequential
from keras.layers.core import Dropout
from keras.layers.convolutional import Convolution2D, GlobalAveragePooling2D
from keras.optimizers import SGD
import cv2, numpy as np

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(Convolution2D(96, 3, 3, border_mode = 'same', input_shape=(3, 32, 32)))
    model.add(Activation('relu'))
    model.add(Convolution2D(96, 3, 3,border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(96, 3, 3, border_mode='same', subsample = (2,2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(192, 3, 3, border_mode = 'same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(192, 3, 3,border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(192, 3, 3,border_mode='same', subsample = (2,2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(192, 3, 3, border_mode = 'same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(192, 1, 1,border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(10, 1, 1, border_mode='valid'))

    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))
    model = make_parallel(model, 4)

    if weights_path:
        model.load_weights(weights_path)

    return model

if __name__ == "__main__":
    im = cv2.resize(cv2.imread('image.jpg'), (224, 224)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)

    # Test pretrained model
    model = VGG_16('weights.994-0.56.hdf5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    out = model.predict(im)
    print np.argmax(out)