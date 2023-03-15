import numpy
import tensorflow as tf
import sys
from tensorflow import keras
from keras import layers
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import libs.ds_layer                # Dempster-Shafer layer
import libs.utility_layer_train     # Utility layer for training
import libs.utility_layer_test      # Utility layer for training
import libs.AU_imprecision          # Metric average utility for set-valued classification
from scipy.optimize import minimize
import math
import numpy as np
import pandas as pd

def probablistic_FitNet4():

    return

if __name__ == '__main__':
    df = pd.read_csv('dataset/KDDCUP99/Encoded/121_100.csv', header=None)
    dataND = df.values
    # print(dataND.shape)
    data2D = dataND.reshape(dataND.shape[0], 10, 10)
    print(data2D[5,:])
    # print(data2D.shape)



