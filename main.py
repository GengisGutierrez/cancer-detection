import tensorflow as tf

import os
import pathlib
import time
import datetime

from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from IPython import display

BUFFER_SIZE = 20
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMAGE_SHAPE = (IMG_WIDTH, IMG_HEIGHT)

ROOT_PATH = "C:/Users/G6367033/PycharmProjects/cancerdetection/dataset/LIDC-IDRI"
TRAIN_PATH = os.path.join(ROOT_PATH, "train")
VAL_PATH = os.path.join(ROOT_PATH, "val")
TEST_PATH = os.path.join(ROOT_PATH, "test")


def ingest_data():
    train_datagen = ImageDataGenerator()
    val_datagen = ImageDataGenerator()
    print("Training images:")
    train_data = train_datagen.flow_from_directory(TRAIN_PATH,
                                                   target_size=IMAGE_SHAPE,
                                                   batch_size=BATCH_SIZE,
                                                   class_mode="input",
                                                   )
    print("Validation images:")
    val_data = val_datagen.flow_from_directory(VAL_PATH,
                                               target_size=IMAGE_SHAPE,
                                               batch_size=BATCH_SIZE,
                                               class_mode="input",
                                               )
    return train_data, val_data


def load(input_image, real_image):
    input_image = tf.io.read_file(input_image)
    input_image = tf.io.decode_jpeg(input_image)
    real_image = tf.io.read_file(real_image)
    real_image = tf.io.decode_jpeg(real_image)
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)
    return input_image, real_image


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, real_image


def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3]
    )
    return cropped_image[0], cropped_image[1]


def load_image_train(train_data):
    train_image, train_label = train_data.next()
    train_label, train_image = random_jitter(train_label[0], train_image[0])
    train_label, train_image = normalize(train_label, train_image)
    return train_image, train_label


def load_image_val(val_data):
    val_image, val_label = val_data.next()
    val_label, val_image = resize(val_label[0], val_image[0],
                                  IMG_HEIGHT, IMG_WIDTH)
    val_label, val_image = normalize(val_label, val_image)
    return val_image, val_label


@tf.function()
def random_jitter(input_image, real_image):
    input_image, real_image = resize(input_image, real_image, 286, 286)
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


def main():
    train_data, val_data = ingest_data()
    train_image, train_label = load_image_train(train_data)
    val_image, val_label = load_image_val(val_data)



if __name__ == "__main__":
    main()
