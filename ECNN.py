import numpy as np
import tensorflow as tf
import traceback
import sys
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from keras import models
from keras import layers
from keras.models import load_model
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

from libs import ds_layer               # Dempster-Shafer layer
from libs import utility_layer_train    # Utility layer for training
from libs import utility_layer_test     # Utility layer for testing
from libs import AU_imprecision         # Metric average utility for set-valued classification
from scipy.optimize import minimize
import math
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
                                                      schedule_decay=0.004), loss='CategoricalCrossentropy',metrics=['accuracy'])
    # model_PR.summary()
    filepath = ''
    filepath1 = ''
    checkpoint_callback = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,save_best_only=True,
                                          save_weights_only=True,save_frequency=1)

    if is_train == 1:
        h = model_PR.fit(x_train, y_train, batch_size=64, epochs=120,
                 verbose=1, callbacks=[checkpoint_callback], validation_data=(x_test, y_test), shuffle=True)
        # model_PR.save_weights(filepath1)
        history = h.history
        epochs = range(len(history['loss']))
        plt.plot(epochs, history['loss'],'b',label='Train loss')
        plt.plot(epochs, history['val_loss'],'r',label='Valid loss')    # Test loss
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.rcParams["figure.dpi"] = 300
        plt.legend()
        plt.savefig('pic/PR-FitNet4-epoch-loss(0.001-64-).png', dpi=300)
        plt.show()

        plt.plot(epochs, history['accuracy'],'b', label='Train accuracy')
        plt.plot(epochs, history['val_accuracy'],'r', label='Valid accuracy')   # Test loss
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.rcParams["figure.dpi"] = 300
        plt.legend()
        plt.savefig('pic/PR-FitNet4-epoch-acc(lr0.01bs64).png', dpi=300)
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
            # print(y_pred)
            # print(y_test)
            # print(type(y_pred))
            # print(type(y_test))
            # print(y_test.shape)
            # print(y_pred.shape)
            confusion_mtx = confusion_matrix(y_test,y_pred)
            print(confusion_mtx)


def evidential_FitNet4(data_WIDTH, data_HEIGHT, num_class, prototypes,is_load,
                       load_weights, train_DS_Layers, plot_DS_layer_loss,
                       x_train, y_train, x_test, y_test,
                       output_confusion_matrix):
    filepath = 'pickleJar/Evidential/EVI_FitNet_0.001and64_500proto'
    inputs = tf.keras.layers.Input((data_WIDTH, data_HEIGHT, 1))
    # convolution stages
    c1_1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1_2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1_1)
    c1_3 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1_2)
    c1_4 = tf.keras.layers.Conv2D(48, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1_3)
    c1_5 = tf.keras.layers.Conv2D(48, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1_4)
    bt1 = tf.keras.layers.BatchNormalization()(c1_5)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(bt1)
    dr1 = tf.keras.layers.Dropout(0.5)(p1)

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

    # DS layer
    ED = ds_layer.DS1(prototypes, 128)(flatten1)
    ED_ac = ds_layer.DS1_activate(prototypes)(ED)
    mass_prototypes = ds_layer.DS2(prototypes, num_class)(ED_ac)
    mass_prototypes_omega = ds_layer.DS2_omega(prototypes, num_class)(mass_prototypes)
    mass_Dempster = ds_layer.DS3_Dempster(prototypes, num_class)(mass_prototypes_omega)
    mass_Dempster_normalize = ds_layer.DS3_normalize()(mass_Dempster)

    # Utility layer for training
    outputs = utility_layer_train.DM(0.9,num_class)(mass_Dempster_normalize)
    model_evi = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model_evi.compile(
        optimizer=keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004),
        loss='CategoricalCrossentropy',
        metrics=['accuracy'])
    # model_evi.summary()

    if load_weights == 1:
        # load the weights of probabilistic classifier
        # and get the feature for training DS layers
        model_evi.load_weights('pickleJar/Probabilistic/PR_FitNet_lr0.01_bs64').expect_partial()
        feature = tf.keras.Model(inputs=[inputs],outputs=[flatten1])
        train_feature_for_DS = feature.predict(x_train)
        test_feature_for_DS = feature.predict(x_test)

    if train_DS_Layers == 1:
        # training DS layers
        inputss = tf.keras.layers.Input(128)
        ED = ds_layer.DS1(prototypes, 128)(inputss)
        ED_ac = ds_layer.DS1_activate(prototypes)(ED)
        mass_prototypes = ds_layer.DS2(prototypes, num_class)(ED_ac)
        mass_prototypes_omega = ds_layer.DS2_omega(prototypes, num_class)(mass_prototypes)
        mass_Dempster = ds_layer.DS3_Dempster(prototypes, num_class)(mass_prototypes_omega)
        mass_Dempster_normalize = ds_layer.DS3_normalize()(mass_Dempster)
        outputss = utility_layer_train.DM(0.9, num_class)(mass_Dempster_normalize)
        model_mid = tf.keras.Model(inputs=[inputss], outputs=[outputss])
        model_mid.compile(
            optimizer=keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004),
            # 0.001
            loss='CategoricalCrossentropy',
            metrics=['accuracy'])
        # model_mid.summary()
        h = model_mid.fit(train_feature_for_DS, y_train, batch_size=64, epochs=2, verbose=1,
                      validation_data=(test_feature_for_DS, y_test), shuffle=True)

        if plot_DS_layer_loss == 1:
            history = h.history
            epochs = range(len(history['loss']))
            plt.plot(epochs, history['loss'], label='Train loss')
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.rcParams["figure.dpi"] = 300
            plt.legend()
            plt.savefig('pic/training loss of DS layer.png', dpi=300)
            plt.show()

        # feed the trained paramters to the evidential model
        model_evi.load_weights('pickleJar/Probabilistic/PR_FitNet_lr0.01_bs64').expect_partial()
        DS1_W = tf.reshape(model_mid.layers[1].get_weights()[0], [1, prototypes, 128])
        DS1_activate_W = model_mid.layers[2].get_weights()
        DS2_W = model_mid.layers[3].get_weights()
        model_evi.layers[26].set_weights(DS1_W)
        model_evi.layers[27].set_weights(DS1_activate_W)
        model_evi.layers[28].set_weights(DS2_W)

        checkpoint_callback = ModelCheckpoint(
            filepath, monitor='val_accuracy', verbose=1,
            save_best_only=True, save_weights_only=True,
            save_frequency=1)
        model_evi.fit(x_train,y_train,
                      batch_size=64, epochs=4, verbose=1,
                      callbacks=[checkpoint_callback], validation_data=(x_test, y_test), shuffle=True)

    if is_load == 1:

        model_evi.load_weights(filepath).expect_partial()
        model_evi.evaluate(x_train, y_train, batch_size=64, verbose=1)
        model_evi.evaluate(x_test, y_test, batch_size=64, verbose=1)
        if output_confusion_matrix == 1:
            y_pred = model_evi.predict(x_test)
            y_pred = y_pred.argmax(axis=1)
            y_test = y_test.argmax(axis=1)
            confusion_mtx = confusion_matrix(y_test,y_pred)
            print(confusion_mtx)


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

    # probabilistic_FitNet4(data_WIDTH=10, data_HEIGHT=10, num_class=5,
    #                       is_train=1, is_load=0,output_confusion_matrix=0,
    #                       x_train=train2D, y_train=train_label,
    #                       x_test=test2D, y_test=test_label)

    evidential_FitNet4(data_WIDTH=10, data_HEIGHT=10, num_class=5,prototypes=500, is_load=1,
                       load_weights=0, train_DS_Layers=0, plot_DS_layer_loss=0,
                       output_confusion_matrix=1,
                       x_train=train2D, y_train=train_label,
                       x_test=test2D, y_test=test_label)

