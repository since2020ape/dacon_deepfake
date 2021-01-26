from glob import glob
from tensorflow.keras import losses

from tensorflow.keras.models import Model, load_model

from tensorflow.keras.layers import Input, Flatten, Dense, concatenate, Dropout, BatchNormalization
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.framework import ops

import tensorflow as tf
import matplotlib.pyplot as plt

## for visualizing
import matplotlib.pyplot as plt, numpy as np

from tqdm import tqdm_notebook

import numbers

from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import nn
from tensorflow.python.framework import ops

import data_generator_dacon as data_generator
import triplet_loss_dacon as triplet_loss

from sklearn.svm import SVC
import csv

import pickle
import sys
import argparse

server_flag = True
model_load_flag = False

regu_weight = 1e-4

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str,
                        help='Path of learning data directory ')
    parser.add_argument ('network_model_path', type=str,
                         help='output path of network model hdf5 ')
    parser.add_argument ('svm_model_path', type=str,
                         help='output path of classifier pickle ')

    return parser.parse_args(argv)


def l2_regularizer(scale, scope='l2_regularizer'):
    """Returns a function that can be used to apply L2 regularization to weights.

    Small values of L2 can help prevent overfitting the training data.

    Args:
    scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.
    scope: An optional scope name.

    Returns:
    A function with signature `l2(weights)` that applies L2 regularization.

    Raises:
    ValueError: If scale is negative or if scale is not a float.
    """
    if isinstance (scale, numbers.Integral):
        raise ValueError ('scale cannot be an integer: %s' % (scale,))
    if isinstance (scale, numbers.Real):
        if scale < 0.:
            raise ValueError ('Setting a scale less than 0 on a regularizer: %g.' %
                              scale)
    if scale == 0.:
        logging.info ('Scale of 0 disables regularizer.')
        return lambda _: None

    def l2(weights):
        """Applies l2 regularization to weights."""
        with ops.name_scope (scope, 'l2_regularizer', [weights]) as name:
            my_scale = ops.convert_to_tensor (scale,
                                              dtype=weights.dtype.base_dtype,
                                              name='scale')
        return standard_ops.multiply (my_scale, nn.l2_loss (weights), name=name)

    return l2


model = None
svm_path = None
test_data = None

class On_Epoch_End_Callback (tf.keras.callbacks.Callback):
    def __init__(self, callbacks):
        self.callbacks = callbacks

    def on_epoch_end(self, epoch, logs=None):
        global model, svm_path
        dacon_leaderboard (model, test_data, epoch, svm_path)
        # print ("fake cnt : %d diff : %d epoch : %d " % (fake_, diff_, epoch))
        # for callback in self.callbacks:
        #     callback(self.model)


def dacon_leaderboard(model, test_data, epoch, svm_path):
    svm_ep = 2
    batch_size = test_data.batch_size

    x_embeddings = np.zeros ((test_data.size_of_sample, 512), np.float32)
    y_embeddings = np.zeros ((test_data.size_of_sample,), int)

    for it in tqdm_notebook (range (test_data.step_per_epoch)):
        x_train, y_train = test_data.__getitem__ (it)
        x_embeddings[it * batch_size: (it + 1) * batch_size] = model.predict (x_train)
        y_embeddings[it * batch_size: (it + 1) * batch_size] = y_train

    ep_x_embeddings = np.zeros ((test_data.size_of_sample * svm_ep, 512), np.float32)
    ep_y_embeddings = np.zeros ((test_data.size_of_sample * svm_ep,), int)

    for it in range (svm_ep):
        ep_x_embeddings[it * test_data.size_of_sample: (it + 1) * test_data.size_of_sample] = x_embeddings
        ep_y_embeddings[it * test_data.size_of_sample: (it + 1) * test_data.size_of_sample] = y_embeddings

    svc_model = SVC (kernel='linear', probability=True)
    svc_model.fit (ep_x_embeddings, ep_y_embeddings)

    svm_pickle_path = (svm_path + '/ep%d_.pkl' % (epoch))

    del x_embeddings, y_embeddings, ep_x_embeddings, ep_y_embeddings

    with open (svm_pickle_path, 'wb') as outfile:
        pickle.dump ((svc_model), outfile)


    return 1


def train_dacon(args):
    global model, svm_path, test_data

    input_shape = (380, 380, 3)
    if server_flag == True:
        batch_size = 28
    else:
        batch_size = 2

    step_per_epoch = 300

    learning_rate = 0.01
    epochs = 1000

    if server_flag == True:

        # 380 380 3
        fake_paths = glob (args.data_path + '/fake/*/*/*/*/*/*.jpg')
        real_paths = glob (args.data_path + '/real/*/*/*/*/*.jpg')

        val_fake_paths = glob (args.data_path+'/validation/fake/*/*.jpg')
        val_real_paths = glob (args.data_path+'/validation/real/*/*.jpg')

    else:

        fake_paths = glob ('Z:/dataset/dacon/deepfake/cropface_380/fake/CW/20200819/*/*/*/*.jpg')
        real_paths = glob ('Z:/dataset/dacon/deepfake/cropface_380/real/CW/20201012/16370/*/*.jpg')

        val_fake_paths = glob ('Z:/dataset/dacon/deepfake/cropface_380/validation/fake/*/*.jpg')
        val_real_paths = glob ('Z:/dataset/dacon/deepfake/cropface_380/validation/real/*/*.jpg')

    train_paths = fake_paths + real_paths
    test_paths = val_fake_paths + val_real_paths
    val_steps = len (test_paths) // batch_size
    print (len (train_paths))
    train_data = data_generator.DataGenerator (train_paths, train_paths, batch_size, step_per_epoch, input_shape, True,
                                               True)
    test_data = data_generator.DataGenerator (test_paths, test_paths, batch_size, val_steps, input_shape, True, False)

    # base_network = tf.keras.applications.EfficientNetB4 (input_shape=input_shape, weights=None, include_top=False, pooling='avg')
    input_images = Input (shape=input_shape, name='input_image')  # input layer for images
    base_network = tf.keras.applications.EfficientNetB4 (input_shape=input_shape, input_tensor=input_images,
                                                         weights='imagenet', include_top=False, pooling='avg')

    x = Dropout (0.4) (base_network.output)
    # x = Dense (128, activation='relu') (x)
    x = Dense (512, kernel_regularizer=l2_regularizer (regu_weight), activation=None) (x)
    # x = Dense (512, activation=None) (x)
    # x = BatchNormalization(scale=False) (x)
    outputs = tf.nn.l2_normalize (x, 1, 1e-10)

    # outputs = Dense (1, activation='sigmoid', name='predictions') (x)
    # outputs = Dense (2, activation='softmax', name='predictions') (x)


    svm_path = args.svm_model_path
    data_path = args.data_path
    model = Model (inputs=input_images, outputs=outputs)

    if model_load_flag == True:

        load_model_path = '../dacon_ckpt/triplet_effB4_ep11_los0.00_BS28.hdf5'
        load_model = tf.keras.models.load_model (load_model_path, custom_objects={
            'triplet_loss_adapted_from_tf': triplet_loss.triplet_loss_adapted_from_tf})

        for layer_target, layer_source in zip (model.layers, load_model.layers):
            weights = layer_source.get_weights ()
            layer_target.set_weights (weights)
            del weights

        del load_model

    # outputs =
    # outputs = base_network ([input_images])

    # opt = Adam (lr=learning_rate)
    # opt = SGD (lr=learning_rate, momentum=0.9)
    opt = SGD (lr=learning_rate)

    # model.compile (loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    # model.compile (loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile (loss=triplet_loss.triplet_loss_adapted_from_tf, optimizer=opt)
    # tf.keras.losses.

    filepath = (args.network_model_path + "/triplet_effB4_ep{epoch:02d}_BS%d.hdf5" % (batch_size))

    # checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, mode='max', save_best_only=True, period=20)
    # checkpoint = ModelCheckpoint (filepath, monitor='val_loss', verbose=1, mode='min', save_best_only=False, period=5)
    checkpoint2 = ModelCheckpoint (filepath, verbose=1, period=1)
    # callbacks_list = [checkpoint, checkpoint2, On_Epoch_End_Callback(None)]
    callbacks_list = [checkpoint2, On_Epoch_End_Callback (None)]

    model.fit_generator (
        generator=train_data,
        validation_data=test_data,
        epochs=epochs,
        callbacks=callbacks_list,
    )


if __name__ == '__main__':
    print(sys.argv[1:])
    train_dacon (parse_arguments(sys.argv[1:]))


