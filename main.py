import tensorflow as tf

import os
import time
import datetime

from matplotlib import pyplot as plt
from IPython import display
from data_ingest import input_pipeline

BUFFER_SIZE = 20
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

ROOT_PATH = "C:/Users/G6367033/PycharmProjects/cancerdetection/dataset/LIDC-IDRI"
TRAIN_PATH = os.path.join(ROOT_PATH, "train")
VAL_PATH = os.path.join(ROOT_PATH, "val")
TEST_PATH = os.path.join(ROOT_PATH, "test")


def main():
    train_dataset, val_dataset = input_pipeline(TRAIN_PATH, VAL_PATH)



if __name__ == "__main__":
    main()
