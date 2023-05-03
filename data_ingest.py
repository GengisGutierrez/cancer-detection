import os
import tensorflow as tf

IMG_WIDTH = 256
IMG_HEIGHT = 256
BUFFER_SIZE = 20
BATCH_SIZE = 1

ROOT_PATH = "C:/Users/G6367033/PycharmProjects/cancerdetection/dataset/LIDC-IDRI"
TRAIN_PATH = os.path.join(ROOT_PATH, "train")
VAL_PATH = os.path.join(ROOT_PATH, "val")
TEST_PATH = os.path.join(ROOT_PATH, "test")


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
    input_image = (input_image / 255.0)
    real_image = (real_image / 255.0)

    return input_image, real_image


def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3]
    )
    return cropped_image[0], cropped_image[1]


def load_image_train(input_image, real_image):
    train_label, train_image = load(input_image, real_image)
    train_label, train_image = random_jitter(train_label, train_image)
    train_label, train_image = normalize(train_label, train_image)
    return train_image, train_label


def load_image_val(input_image, real_image):
    val_label, val_image = load(input_image, real_image)
    val_label, val_image = resize(val_label, val_image,
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


def input_pipeline(train_path=TRAIN_PATH, val_path=VAL_PATH):
    train_feature_dataset = tf.data.Dataset.list_files(os.path.join(train_path, "feature/*.jpg"))
    train_label_dataset = tf.data.Dataset.list_files(os.path.join(train_path, "label/*.jpg"))

    train_dataset = tf.data.Dataset.zip((train_feature_dataset, train_label_dataset))
    train_dataset = train_dataset.map(lambda x, y: (load_image_train(y, x)),
                                      num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    val_feature_dataset = tf.data.Dataset.list_files(os.path.join(val_path, "feature/*.jpg"))
    val_label_dataset = tf.data.Dataset.list_files(os.path.join(val_path, "label/*jpg"))

    val_dataset = tf.data.Dataset.zip((val_feature_dataset, val_label_dataset))
    val_dataset = val_dataset.map(lambda x, y: (load_image_val(y, x)))
    val_dataset = val_dataset.batch(BATCH_SIZE)

    return train_dataset, val_dataset
