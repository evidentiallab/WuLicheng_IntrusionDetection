import numpy as np
import tensorflow as tf
import traceback
import sys
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
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


def probabilistic_conv2maxpool2(data_WIDTH, data_HEIGHT, num_class,
                                is_train, is_load, model_filepath,
                                pic_filepath_loss, pic_filepath_acc,
                                x_train, y_train, x_test, y_test,
                                output_confusion_matrix):
    inputs = tf.keras.layers.Input((data_WIDTH, data_HEIGHT, 1))

    c1_1 = keras.layers.Conv2D(8, (2, 2), activation='relu', kernel_initializer='he_normal', padding='same',strides=1)(inputs)
    bt1 = keras.layers.BatchNormalization()(c1_1)
    p1 = keras.layers.MaxPooling2D((2, 2),strides=1)(bt1)
    # dr1 = keras.layers.Dropout(0.5)(p1)

    c2_1 = keras.layers.Conv2D(16, (2, 2), activation='relu', kernel_initializer='he_normal', padding='same',strides=1)(p1)
    p2 = keras.layers.MaxPooling2D((2, 2),strides=1)(c2_1)
    dr2 = keras.layers.Dropout(0.5)(p2)

    flatten1 = keras.layers.Flatten()(dr2)
    d1 = keras.layers.Dense(64,activation='relu')(flatten1)
    dr3 = keras.layers.Dropout(0.5)(d1)
    outputs = tf.keras.layers.Dense(num_class, activation='softmax')(dr3)

    model_PR = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model_PR.compile(optimizer=keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
                                                      schedule_decay=0.004), loss='CategoricalCrossentropy',metrics=['accuracy'])

    # model_PR.summary()
    checkpoint_callback = ModelCheckpoint(model_filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                          save_weights_only=True, save_frequency=1)
    if is_train == 1:
        h = model_PR.fit(x_train, y_train, batch_size=32, epochs=60,
                         verbose=1, callbacks=[checkpoint_callback], validation_data=(x_test, y_test), shuffle=True)
        history = h.history
        epochs = range(len(history['loss']))
        plt.plot(epochs, history['loss'], 'royalblue', label='Train loss')
        plt.plot(epochs, history['val_loss'], 'crimson', label='Valid loss')  # Test loss
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.rcParams["figure.dpi"] = 300
        plt.legend()
        plt.savefig(pic_filepath_loss, dpi=300)
        # plt.show()
        plt.close()

        plt.plot(epochs, history['accuracy'], 'royalblue', label='Train accuracy')
        plt.plot(epochs, history['val_accuracy'], 'crimson', label='Valid accuracy')  # Test loss
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.rcParams["figure.dpi"] = 300
        plt.legend()
        plt.savefig(pic_filepath_acc, dpi=300)
        # plt.show()
        plt.close()
    if is_load == 1:
        model_PR.load_weights(model_filepath).expect_partial()
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
            confusion_mtx = confusion_matrix(y_test, y_pred)
            normalized_confusion_mtx = np.round(confusion_matrix(y_test, y_pred, normalize='true'), 3)

            f1 = f1_score(y_true=y_test, y_pred=y_pred, labels=None, average='macro')
            acc = accuracy_score(y_true=y_test, y_pred=y_pred)
            recall = recall_score(y_true=y_test, y_pred=y_pred, labels=None, average='macro')
            preccision = precision_score(y_true=y_test, y_pred=y_pred, labels=None, average='macro')
            print(confusion_mtx)
            print(normalized_confusion_mtx)
            print('accuracy: ' + str(acc))
            print('recall: ' + str(recall))
            print('precision: ' + str(preccision))
            print('F1-score: ' + str(f1))


def evidential_conv2maxpool2(data_WIDTH, data_HEIGHT, num_class, prototypes,nu,model_filepath,evi_filepath,mid_filepath,
                       is_load,load_and_train, plot_DS_layer_loss,
                       x_train, y_train, x_test, y_test,
                       output_confusion_matrix):

    inputs = tf.keras.layers.Input((data_WIDTH, data_HEIGHT, 1))
    # convolution stages
    c1_1 = keras.layers.Conv2D(8, (2, 2), activation='relu', kernel_initializer='he_normal', padding='same', strides=1)(inputs)
    bt1 = keras.layers.BatchNormalization()(c1_1)
    p1 = keras.layers.MaxPooling2D((2, 2), strides=1)(bt1)
    # dr1 = keras.layers.Dropout(0.5)(p1)

    c2_1 = keras.layers.Conv2D(16, (2, 2), activation='relu', kernel_initializer='he_normal', padding='same',strides=1)(p1)
    p2 = keras.layers.MaxPooling2D((2, 2), strides=1)(c2_1)
    dr2 = keras.layers.Dropout(0.5)(p2)
    flatten1 = keras.layers.Flatten()(dr2)
    d1 = keras.layers.Dense(64,activation='relu')(flatten1)

    # DS layer
    ED = ds_layer.DS1(prototypes, 64)(d1)
    ED_ac = ds_layer.DS1_activate(prototypes)(ED)
    mass_prototypes = ds_layer.DS2(prototypes, num_class)(ED_ac)
    mass_prototypes_omega = ds_layer.DS2_omega(prototypes, num_class)(mass_prototypes)
    mass_Dempster = ds_layer.DS3_Dempster(prototypes, num_class)(mass_prototypes_omega)
    mass_Dempster_normalize = ds_layer.DS3_normalize()(mass_Dempster)

    # Utility layer for training
    outputs = utility_layer_train.DM(nu,num_class)(mass_Dempster_normalize)
    model_evi = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model_evi.compile(
        optimizer=keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004),
        loss='CategoricalCrossentropy',
        metrics=['accuracy'])
    # model_evi.summary()

    if load_and_train == 1:
        # load the weights of probabilistic CNN
        # and get the feature for training DS layers
        model_evi.load_weights(model_filepath).expect_partial()
        feature = tf.keras.Model(inputs=[inputs],outputs=[d1])
        train_feature_for_DS = feature.predict(x_train)
        test_feature_for_DS = feature.predict(x_test)

        # training DS layers
        inputss = tf.keras.layers.Input(64)
        ED = ds_layer.DS1(prototypes, 64)(inputss)
        ED_ac = ds_layer.DS1_activate(prototypes)(ED)
        mass_prototypes = ds_layer.DS2(prototypes, num_class)(ED_ac)
        mass_prototypes_omega = ds_layer.DS2_omega(prototypes, num_class)(mass_prototypes)
        mass_Dempster = ds_layer.DS3_Dempster(prototypes, num_class)(mass_prototypes_omega)
        mass_Dempster_normalize = ds_layer.DS3_normalize()(mass_Dempster)
        outputss = utility_layer_train.DM(nu, num_class)(mass_Dempster_normalize)
        model_mid = tf.keras.Model(inputs=[inputss], outputs=[outputss])
        model_mid.compile(
            optimizer=keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004),
            # 0.001
            loss='CategoricalCrossentropy',metrics=['accuracy'])
        # model_mid.summary()
        mid_callback = ModelCheckpoint(mid_filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                       save_weights_only=True, save_frequency=1)
        h = model_mid.fit(train_feature_for_DS, y_train, batch_size=64, epochs=10, verbose=1,
                      validation_data=(test_feature_for_DS, y_test), callbacks=[mid_callback],shuffle=True)

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

        # feed the trained paramters to the evidential disusedModel
        model_evi.load_weights(model_filepath).expect_partial()
        DS1_W = tf.reshape(model_mid.layers[1].get_weights()[0], [1, prototypes, 64])
        DS1_activate_W = model_mid.layers[2].get_weights()
        DS2_W = model_mid.layers[3].get_weights()
        model_evi.layers[9].set_weights(DS1_W)
        model_evi.layers[10].set_weights(DS1_activate_W)
        model_evi.layers[11].set_weights(DS2_W)

        checkpoint_callback = ModelCheckpoint(
            evi_filepath, monitor='val_accuracy', verbose=1,
            save_best_only=True, save_weights_only=True,
            save_frequency=1)
        model_evi.fit(x_train,y_train,
                      batch_size=64, epochs=8, verbose=1,
                      callbacks=[checkpoint_callback], validation_data=(x_test, y_test), shuffle=True)

    if is_load == 1:
        model_evi.load_weights(evi_filepath).expect_partial()
        model_evi.evaluate(x_train, y_train, batch_size=64, verbose=1)
        model_evi.evaluate(x_test, y_test, batch_size=64, verbose=1)
        if output_confusion_matrix == 1:
            y_pred = model_evi.predict(x_test)
            y_pred = y_pred.argmax(axis=1)
            y_test = y_test.argmax(axis=1)
            confusion_mtx = confusion_matrix(y_test, y_pred)
            f1 = f1_score(y_true=y_test, y_pred=y_pred, labels=None, average='macro')
            acc = accuracy_score(y_true=y_test, y_pred=y_pred)
            recall = recall_score(y_true=y_test, y_pred=y_pred, labels=None, average='macro')
            preccision = precision_score(y_true=y_test, y_pred=y_pred, labels=None, average='macro')
            # print(confusion_mtx)
            # print(np.round(confusion_matrix(y_test, y_pred,normalize='true'),3))
            print('proto: ' + str(prototypes) + 'nu: ' + str(nu))
            print('accuracy: ' + str(acc))
            print('recall: ' + str(recall))
            print('precision: ' + str(preccision))
            print('F1-score: ' + str(f1))
            return acc