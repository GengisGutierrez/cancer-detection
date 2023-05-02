import tensorflow as tf

import os
import time
import datetime

from matplotlib import pyplot as plt
from IPython import display
from data_ingest import input_pipeline

OUTPUT_CHANNELS = 3


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result




def main():
    global img
    train_dataset, val_dataset = input_pipeline()
    for train_image, train_label in train_dataset.take(1):
        img = train_image[0].numpy()
        lbl = train_label[0].numpy()

    down_model = downsample(3, 4)
    down_result = down_model(tf.expand_dims(img, 0))
    up_model = upsample(3, 4)
    up_result = up_model(down_result)
    print(up_result.shape)


if __name__ == "__main__":
    main()
