import numpy
import tensorflow as tf
import sys
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from keras import models
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
import matplotlib.pyplot as plt
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def probabilistic_FitNet4(data_WIDTH, data_HEIGHT, num_class,
                          is_train, is_load,
                          x_train, y_train, x_test, y_test,
                          output_confusion_matrix):
    inputs = tf.keras.layers.Input((data_WIDTH,data_HEIGHT,1))

    c1_1 = keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1_2 = keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1_1)
    c1_3 = keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1_2)
    c1_4 = keras.layers.Conv2D(48, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1_3)
    c1_5 = keras.layers.Conv2D(48, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1_4)
    bt1 = keras.layers.BatchNormalization()(c1_5)
    p1 = keras.layers.MaxPooling2D((2, 2))(bt1)
    dr1 = keras.layers.Dropout(0.5)(p1)

    c2_1 = tf.keras.layers.Conv2D(80, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(dr1)
    c2_2 = tf.keras.layers.Conv2D(80, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2_1)
    c2_3 = tf.keras.layers.Conv2D(80, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2_2)
    c2_4 = tf.keras.layers.Conv2D(80, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2_3)
    c2_5 = tf.keras.layers.Conv2D(80, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2_4)
    bt2 = tf.keras.layers.BatchNormalization()(c2_5)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(bt2)
    dr2 = tf.keras.layers.Dropout(0.5)(p2)

    c3_1 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(dr2)
    c3_2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3_1)
    c3_3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3_2)
    c3_4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3_3)
    c3_5 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3_4)
    bt3 = tf.keras.layers.BatchNormalization()(c3_5)
    # p3 = tf.keras.layers.MaxPooling2D((8, 8))(bt3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(bt3)
    dr3 = tf.keras.layers.Dropout(0.5)(p3)

    flatten1 = tf.keras.layers.Flatten()(dr3)

    outputs = tf.keras.layers.Dense(num_class, activation='softmax')(flatten1)
    model_PR = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model_PR.compile(optimizer=keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
                                                      schedule_decay=0.004), loss='CategoricalCrossentropy',
                     metrics=['accuracy'])
    # model_PR.summary()
    filepath = 'pickleJar/PR_FitNet4'
    filepath1 = 'pickleJar/PR_FitNet4_best.hdf5'
    checkpoint_callback = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,save_best_only=True,
                                          save_weights_only=True,save_frequency=1)

    if is_train == 1:
        h = model_PR.fit(x_train, y_train, batch_size=32, epochs=120,
                 verbose=1, callbacks=[checkpoint_callback], validation_data=(x_test, y_test), shuffle=True)
        model_PR.save_weights(filepath1)
        history = h.history
        epochs = range(len(history['loss']))
        plt.plot(epochs, history['loss'],'b',label='Train loss')
        plt.plot(epochs, history['val_loss'],'r',label='Valid loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.rcParams["figure.dpi"] = 300
        plt.legend()
        plt.savefig('pic/PR-FitNet4-epoch-loss.png', dpi=300)
        plt.show()

        plt.plot(epochs, history['accuracy'],'b', label='Train accuracy')
        plt.plot(epochs, history['val_accuracy'],'r', label='Valid accuracy')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.rcParams["figure.dpi"] = 300
        plt.legend()
        plt.savefig('pic/PR-FitNet4-epoch-acc.png', dpi=300)
        plt.show()

    if is_load == 1:
        model_PR.load_weights(filepath).expect_partial()
        # model_PR.load_weights(filepath1)
        # model_PR.evaluate(x_train, y_train, batch_size=25, verbose=1)
        # model_PR.evaluate(x_test, y_test, batch_size=32, verbose=1)
        if output_confusion_matrix == 1:
            y_pred = model_PR.predict(x_test)
            y_pred = y_pred.argmax(axis=1)
            y_test = y_test.argmax(axis=1)
            print(y_pred)
            print(y_test)
            # print(type(y_pred))
            # print(type(y_test))
            # print(y_test.shape)
            # print(y_pred.shape)
            confusion_mtx = confusion_matrix(y_test,y_pred)
            print(confusion_mtx)


def evidential_FitNet4(data_WIDTH,data_HEIGHT,num_class,
                         is_train,is_load,
                         x_train,y_train,x_test,y_test,
                         output_confusion_matrix):


if __name__ == '__main__':
    # df = pd.read_csv('dataset/KDDCUP99/Encoded/121_100.csv', header=None)
    # dataND = df.values
    # # print(dataND.shape)
    # data2D = dataND.reshape(dataND.shape[0], 10, 10)

    # 读取训练集
    trainDF = pd.read_csv('dataset/KDDCUP99/Encoded/train(123_100).csv', header=None)
    trainND = trainDF.values
    train2D = trainND.reshape(trainND.shape[0], 10, 10)
    train_labelDF = pd.read_csv('dataset/KDDCUP99/Label/trainLabel/multi-label-onehot.csv', header=0)
    train_label = train_labelDF.values
    # 读取测试集
    testDF = pd.read_csv('dataset/KDDCUP99/Encoded/test(123_100).csv', header=None)
    testND = testDF.values
    test2D = testND.reshape(testND.shape[0], 10, 10)
    test_labelDF = pd.read_csv('dataset/KDDCUP99/Label/testLabel/multi-label-onehot.csv', header=0)
    test_label = test_labelDF.values
    # print('shape of train'+str(train2D.shape))
    # print('shape of train_label' + str(train_label.shape))
    # print('shape of test' + str(test2D.shape))
    # print('shape of test_label' + str(test_label.shape))

    probabilistic_FitNet4(data_WIDTH=10, data_HEIGHT=10, num_class=5,
                          is_train=0, is_load=1,
                          x_train=train2D, y_train=train_label,
                          x_test=test2D, y_test=test_label,
                          output_confusion_matrix=1)

