import numpy as np
import pandas as pd
import pickle
from os import path
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scipy.stats import chi2_contingency
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, LSTM, MaxPool1D, Flatten, Dropout,Conv1D,BatchNormalization
from keras.models import Sequential
from keras.layers import Input
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint
from libs import ds_layer               # Dempster-Shafer layer
from libs import utility_layer_train    # Utility layer for training
from libs import utility_layer_test     # Utility layer for testing
from libs import AU_imprecision         # Metric average utility for set-valued classification
from scipy.optimize import minimize
import math


def probabilistic_triCNN(data_WIDTH,num_class,is_train, is_load, model_filepath,x_train,
                        y_train, x_test, y_test,output_confusion_matrix):
    inputs = keras.layers.Input((data_WIDTH, 1))
    c1 = keras.layers.Conv1D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c2 = keras.layers.Conv1D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    pool1 = keras.layers.MaxPooling1D(3)(c2)
    c3 = keras.layers.Conv1D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same')(pool1)
    pool2 = keras.layers.MaxPooling1D(3)(c3)
    bn1 = keras.layers.BatchNormalization()(pool2)
    dr1 = keras.layers.Dropout(0.5)(bn1)
    flatten1 = tf.keras.layers.Flatten()(dr1)
    outputs = tf.keras.layers.Dense(num_class, activation='softmax')(flatten1)

    model_PR = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model_PR.compile(optimizer=keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
                                                      schedule_decay=0.004), loss='CategoricalCrossentropy',
                     metrics=['accuracy'])
    # model_PR.summary()

    checkpoint_callback = ModelCheckpoint(model_filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                          save_weights_only=True, save_frequency=1)

    if is_train == 1:
        h = model_PR.fit(x_train, y_train, batch_size=32, epochs=240,
                         verbose=1, callbacks=[checkpoint_callback], validation_data=(x_test, y_test), shuffle=True)
        history = h.history
    if is_load == 1:
        model_PR.load_weights(model_filepath).expect_partial()
        # model_PR.load_weights(filepath1)
        # model_PR.evaluate(x_train, y_train, batch_size=25, verbose=1)
        # model_PR.evaluate(x_test, y_test, batch_size=32, verbose=1)
        if output_confusion_matrix == 1:
            y_pred = model_PR.predict(x_test)
            y_pred = y_pred.argmax(axis=1)
            y_test = y_test.argmax(axis=1)
            confusion_mtx = confusion_matrix(y_test, y_pred)
            f1 = f1_score(y_true=y_test, y_pred=y_pred, labels=None, average='macro')
            acc = accuracy_score(y_true=y_test, y_pred=y_pred)
            recall = recall_score(y_true=y_test, y_pred=y_pred, labels=None, average='macro')
            preccision = precision_score(y_true=y_test, y_pred=y_pred, labels=None, average='macro')
            print(confusion_mtx)
            print('accuracy: ' + str(acc))
            print('recall: ' + str(recall))
            print('precision: ' + str(preccision))
            print('F1-score: ' + str(f1))


def probabilistic_biCNN(data_WIDTH,num_class,is_train, is_load, model_filepath,x_train,
                        y_train, x_test, y_test,output_confusion_matrix):
    inputs = keras.layers.Input((data_WIDTH, 1))
    c1 = keras.layers.Conv1D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c2 = keras.layers.Conv1D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    pool1 = keras.layers.MaxPooling1D(3)(c2)
    bn1 = keras.layers.BatchNormalization()(pool1)
    dr1 = keras.layers.Dropout(0.5)(bn1)
    flatten1 = tf.keras.layers.Flatten()(dr1)
    # fc1 = tf.keras.layers.Dense(64,activation='relu')(flatten1)
    outputs = tf.keras.layers.Dense(num_class, activation='softmax')(flatten1)

    model_PR = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model_PR.compile(optimizer=keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
                                                      schedule_decay=0.004), loss='CategoricalCrossentropy',
                     metrics=['accuracy'])
    model_PR.summary()

    checkpoint_callback = ModelCheckpoint(model_filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                          save_weights_only=True, save_frequency=1)

    if is_train == 1:
        h = model_PR.fit(x_train, y_train, batch_size=32, epochs=240,
                         verbose=1, callbacks=[checkpoint_callback], validation_data=(x_test, y_test), shuffle=True)

        history = h.history
        # epochs = range(len(history['loss']))
        # plt.figure()
        # plt.plot(epochs, history['loss'], 'b', label='Train loss')
        # plt.plot(epochs, history['val_loss'], 'r', label='Valid loss')  # Test loss
        # plt.xlabel("Epochs")
        # plt.ylabel("Loss")
        # plt.rcParams["figure.dpi"] = 300
        # plt.legend()
        # plt.savefig(pic_fp0, dpi=300)
        # # plt.show()
        # plt.close()
        #
        # plt.figure()
        # plt.plot(epochs, history['accuracy'], 'b', label='Train accuracy')
        # plt.plot(epochs, history['val_accuracy'], 'r', label='Valid accuracy')  # Test loss
        # plt.xlabel("Epochs")
        # plt.ylabel("accuracy")
        # plt.rcParams["figure.dpi"] = 300
        # plt.legend()
        # plt.savefig(pic_fp1, dpi=300)
        # # plt.show()
        # plt.close()

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
            # confusion_mtx = confusion_matrix(y_test, y_pred)
            f1 = f1_score(y_true=y_test, y_pred=y_pred, labels=None, average='macro')
            acc = accuracy_score(y_true=y_test, y_pred=y_pred)
            recall = recall_score(y_true=y_test, y_pred=y_pred, labels=None, average='macro')
            preccision = precision_score(y_true=y_test, y_pred=y_pred, labels=None, average='macro')
            # print(confusion_mtx)
            print('accuracy: ' + str(acc))
            print('recall: ' + str(recall))
            print('precision: ' + str(preccision))
            print('F1-score: ' + str(f1))


def probabilistic_biCNN_SV(data_WIDTH,num_class,is_train, is_load, model_filepath,x_train,
                        y_train, x_test, y_test,output_confusion_matrix):
    inputs = keras.layers.Input((data_WIDTH, 1))
    c1 = keras.layers.Conv1D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c2 = keras.layers.Conv1D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    pool1 = keras.layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(c2)
    bn1 = keras.layers.BatchNormalization()(pool1)
    dr1 = keras.layers.Dropout(0.2)(bn1)
    flatten1 = tf.keras.layers.Flatten()(dr1)
    fc1 = tf.keras.layers.Dense(64,activation='relu')(flatten1)
    outputs = tf.keras.layers.Dense(num_class, activation='softmax')(fc1)


def evidential_CNN_api(data_WIDTH, num_class, prototypes,nu,model_filepath,evi_filepath,mid_filepath,
                       flatten_size,load_and_train, is_load,
                       x_train, y_train, x_test, y_test,
                       output_confusion_matrix):
    inputs = keras.layers.Input((data_WIDTH, 1))
    c1 = keras.layers.Conv1D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c2 = keras.layers.Conv1D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    pool1 = keras.layers.MaxPooling1D(3)(c2)
    bn1 = keras.layers.BatchNormalization()(pool1)
    dr1 = keras.layers.Dropout(0.2)(bn1)
    flatten1 = tf.keras.layers.Flatten()(dr1)
    # fc1 = tf.keras.layers.Dense(64, activation='relu')(flatten1)

    # DS layer
    ED = ds_layer.DS1(prototypes, flatten_size)(flatten1)
    ED_ac = ds_layer.DS1_activate(prototypes)(ED)
    mass_prototypes = ds_layer.DS2(prototypes, num_class)(ED_ac)
    mass_prototypes_omega = ds_layer.DS2_omega(prototypes, num_class)(mass_prototypes)
    mass_Dempster = ds_layer.DS3_Dempster(prototypes, num_class)(mass_prototypes_omega)
    mass_Dempster_normalize = ds_layer.DS3_normalize()(mass_Dempster)

    # Utility layer for training
    outputs = utility_layer_train.DM(nu, num_class)(mass_Dempster_normalize)
    model_evi = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model_evi.compile(
        optimizer=keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
                                         schedule_decay=0.004),
        loss='CategoricalCrossentropy',
        metrics=['accuracy'])
    # model_evi.summary()

    if load_and_train == 1:
        # load the weights of probabilistic classifier
        # and get the feature for training DS layers
        model_evi.load_weights(model_filepath).expect_partial()
        feature = tf.keras.Model(inputs=[inputs], outputs=[flatten1])
        train_feature_for_DS = feature.predict(x_train)
        test_feature_for_DS = feature.predict(x_test)

        # training DS layers
        inputss = tf.keras.layers.Input(flatten_size)
        ED = ds_layer.DS1(prototypes, flatten_size)(inputss)
        ED_ac = ds_layer.DS1_activate(prototypes)(ED)
        mass_prototypes = ds_layer.DS2(prototypes, num_class)(ED_ac)
        mass_prototypes_omega = ds_layer.DS2_omega(prototypes, num_class)(mass_prototypes)
        mass_Dempster = ds_layer.DS3_Dempster(prototypes, num_class)(mass_prototypes_omega)
        mass_Dempster_normalize = ds_layer.DS3_normalize()(mass_Dempster)
        outputss = utility_layer_train.DM(nu, num_class)(mass_Dempster_normalize)
        model_mid = tf.keras.Model(inputs=[inputss], outputs=[outputss])
        model_mid.compile(
            optimizer=keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
                                             schedule_decay=0.004),loss='CategoricalCrossentropy',metrics=['accuracy'])
        # model_mid.summary()
        mid_callback = ModelCheckpoint(mid_filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                       save_weights_only=True, save_frequency=1)
        model_mid.fit(train_feature_for_DS, y_train, batch_size=64, epochs=20, verbose=1,
                          validation_data=(test_feature_for_DS, y_test), callbacks=[mid_callback], shuffle=True)

        # feed the trained paramters to the evidential disusedModel
        model_evi.load_weights(model_filepath).expect_partial()
        DS1_W = tf.reshape(model_mid.layers[1].get_weights()[0], [1, prototypes, flatten_size])
        DS1_activate_W = model_mid.layers[2].get_weights()
        DS2_W = model_mid.layers[3].get_weights()
        model_evi.layers[7].set_weights(DS1_W)
        model_evi.layers[8].set_weights(DS1_activate_W)
        model_evi.layers[9].set_weights(DS2_W)

        checkpoint_callback = ModelCheckpoint(
            evi_filepath, monitor='val_accuracy', verbose=1,
            save_best_only=True, save_weights_only=True,
            save_frequency=1)
        model_evi.fit(x_train, y_train,
                      batch_size=64, epochs=18, verbose=1,
                      callbacks=[checkpoint_callback], validation_data=(x_test, y_test), shuffle=True)

    if is_load == 1:
        model_evi.load_weights(evi_filepath).expect_partial()
        # model_evi.evaluate(x_train, y_train, batch_size=64, verbose=1)
        # model_evi.evaluate(x_test, y_test, batch_size=64, verbose=1)
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


def evidential_CNN_api_nofinetune(data_WIDTH, num_class, prototypes,nu,model_filepath,mid_filepath,
                       is_train, is_load,
                       x_train, y_train, x_test, y_test,
                       output_confusion_matrix):
    inputs = keras.layers.Input((data_WIDTH, 1))
    c1 = keras.layers.Conv1D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c2 = keras.layers.Conv1D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    pool1 = keras.layers.MaxPooling1D(3)(c2)
    bn1 = keras.layers.BatchNormalization()(pool1)
    dr1 = keras.layers.Dropout(0.5)(bn1)
    flatten1 = tf.keras.layers.Flatten()(dr1)

    # DS layer
    ED = ds_layer.DS1(prototypes, 448)(flatten1)
    ED_ac = ds_layer.DS1_activate(prototypes)(ED)
    mass_prototypes = ds_layer.DS2(prototypes, num_class)(ED_ac)
    mass_prototypes_omega = ds_layer.DS2_omega(prototypes, num_class)(mass_prototypes)
    mass_Dempster = ds_layer.DS3_Dempster(prototypes, num_class)(mass_prototypes_omega)
    mass_Dempster_normalize = ds_layer.DS3_normalize()(mass_Dempster)

    # Utility layer for training
    outputs = utility_layer_train.DM(nu, num_class)(mass_Dempster_normalize)
    model_evi = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model_evi.compile(
        optimizer=keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
                                         schedule_decay=0.004),
        loss='CategoricalCrossentropy',
        metrics=['accuracy'])

    # model_evi.summary()
    # load the weights of probabilistic classifier and get the feature for training DS layers
    model_evi.load_weights(model_filepath).expect_partial()
    feature = tf.keras.Model(inputs=[inputs], outputs=[flatten1])
    train_feature_for_DS = feature.predict(x_train)
    test_feature_for_DS = feature.predict(x_test)
    # training DS layers
    inputss = tf.keras.layers.Input(448)
    ED = ds_layer.DS1(prototypes, 448)(inputss)
    ED_ac = ds_layer.DS1_activate(prototypes)(ED)
    mass_prototypes = ds_layer.DS2(prototypes, num_class)(ED_ac)
    mass_prototypes_omega = ds_layer.DS2_omega(prototypes, num_class)(mass_prototypes)
    mass_Dempster = ds_layer.DS3_Dempster(prototypes, num_class)(mass_prototypes_omega)
    mass_Dempster_normalize = ds_layer.DS3_normalize()(mass_Dempster)
    outputss = utility_layer_train.DM(nu, num_class)(mass_Dempster_normalize)
    model_mid = tf.keras.Model(inputs=[inputss], outputs=[outputss])
    if is_train == 1:
        model_mid.compile(
            optimizer=keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
                                             schedule_decay=0.004),loss='CategoricalCrossentropy',metrics=['accuracy'])
        # model_mid.summary()
        mid_callback = ModelCheckpoint(mid_filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                       save_weights_only=True, save_frequency=1)
        h = model_mid.fit(train_feature_for_DS, y_train, batch_size=64, epochs=30, verbose=1,
                          validation_data=(test_feature_for_DS, y_test), callbacks=[mid_callback], shuffle=True)

    if is_load == 1:
        # feed the trained paramters to the evidential disusedModel
        # model_evi.load_weights(model_filepath).expect_partial()
        DS1_W = tf.reshape(model_mid.layers[1].get_weights()[0], [1, prototypes, 448])
        DS1_activate_W = model_mid.layers[2].get_weights()
        DS2_W = model_mid.layers[3].get_weights()
        model_evi.layers[7].set_weights(DS1_W)
        model_evi.layers[8].set_weights(DS1_activate_W)
        model_evi.layers[9].set_weights(DS2_W)
        # model_evi.evaluate(x_train, y_train, batch_size=64, verbose=1)
        # model_evi.evaluate(x_test, y_test, batch_size=64, verbose=1)
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


def set_value_api(data_WIDTH,num_class, number_act_set, act_set, nu, tol,prototypes,
                  utility_matrix, is_load, filepath,x_test, numerical_y_test):
    inputs = keras.layers.Input((data_WIDTH, 1))
    c1 = keras.layers.Conv1D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c2 = keras.layers.Conv1D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    pool1 = keras.layers.MaxPooling1D(3)(c2)
    bn1 = keras.layers.BatchNormalization()(pool1)
    dr1 = keras.layers.Dropout(0)(bn1)
    flatten1 = tf.keras.layers.Flatten()(dr1)
    # fc1 = tf.keras.layers.Dense(64, activation='relu')(flatten1)
    # DS layer
    ED = ds_layer.DS1(prototypes, 512)(flatten1)
    ED_ac = ds_layer.DS1_activate(prototypes)(ED)
    mass_prototypes = ds_layer.DS2(prototypes, num_class)(ED_ac)
    mass_prototypes_omega = ds_layer.DS2_omega(prototypes, num_class)(mass_prototypes)
    mass_Dempster = ds_layer.DS3_Dempster(prototypes, num_class)(mass_prototypes_omega)
    mass_Dempster_normalize = ds_layer.DS3_normalize()(mass_Dempster)

    # Utility layer for training
    outputs = utility_layer_test.DM_test(num_class, number_act_set, nu)(mass_Dempster_normalize)
    model_evi_SV = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model_evi_SV.compile(
        optimizer=keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
                                         schedule_decay=0.004), loss='CategoricalCrossentropy', metrics=['accuracy'])
    # model_evi_SV.summary()
    if is_load:
        model_evi_SV.layers[-1].set_weights(tf.reshape(utility_matrix, [1, number_act_set, num_class]))
        model_evi_SV.load_weights(filepath).expect_partial()

        results = tf.argmax(model_evi_SV.predict(x_test), -1)
        # print(model_evi_SV.predict(x_test).shape)
        # print(results.shape)
        imprecise_results = []
        sv_counter = 0
        for i in range(len(results)):
            act_local = results[i]
            set_valued_results = act_set[act_local]
            imprecise_results.append(set_valued_results)

            if len(set_valued_results) > 1:
                sv_counter = sv_counter + 1
        # print(imprecise_results)
        average_utility_imprecision = AU_imprecision.average_utility(utility_matrix, results, numerical_y_test, act_set)
        print('prototypes=' + str(prototypes) + ' nu=' + str(nu) + ' tol=' + str(tol))
        print('**** **** **** **** **** **** AU = ' + str(average_utility_imprecision) + ' **** **** **** **** **** *****')
        print('**** **** **** **** **** **** SV Rates assignment = ' + str(
            (sv_counter / len(results)) * 100) + ' **** **** **** **** **** *****')


def bilayer_LSTM_api(hidden_units, num_class, model_filepath,is_train, is_load,
                     x_train, y_train, x_test, y_test, pic_filepath_loss, pic_filepath_acc,output_confusion_matrix):

    inputs = Input((x_train.shape[1], x_train.shape[2]))
    lstm1 = LSTM(units=hidden_units, return_sequences=True,dropout=0,recurrent_dropout=0)(inputs)
    lstm2 = LSTM(units=hidden_units, return_sequences=True,dropout=0,recurrent_dropout=0)(lstm1)
    flatten1 = Flatten()(lstm2)
    outputs = Dense(units=num_class, activation='softmax')(flatten1)
    model = keras.Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer=keras.optimizers.Nadam(learning_rate=0.001,beta_1=0.9, beta_2=0.999, epsilon=None,
                                                   schedule_decay=0.004), loss='CategoricalCrossentropy',metrics=['accuracy'])

    # disusedModel.summary()

    checkpoint_callback = ModelCheckpoint(model_filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                          save_weights_only=True, save_frequency=1)
    if is_train == 1:
        h = model.fit(x_train, y_train, batch_size=32, epochs=120,
                      verbose=1, callbacks=[checkpoint_callback], validation_data=(x_test, y_test), shuffle=True)
        history = h.history
        # epochs = range(len(history['loss']))
        # plt.plot(epochs, history['loss'], 'royalblue', label='Train loss')
        # plt.plot(epochs, history['val_loss'], 'crimson', label='Valid loss')  # Test loss
        # plt.xlabel("Epochs")
        # plt.ylabel("Loss")
        # plt.rcParams["figure.dpi"] = 300
        # plt.legend()
        # plt.savefig(pic_filepath_loss, dpi=300)
        # plt.close()
        #
        # plt.plot(epochs, history['accuracy'], 'royalblue', label='Train accuracy')
        # plt.plot(epochs, history['val_accuracy'], 'crimson', label='Valid accuracy')  # Test loss
        # plt.xlabel("Epochs")
        # plt.ylabel("Accuracy")
        # plt.rcParams["figure.dpi"] = 300
        # plt.legend()
        # plt.savefig(pic_filepath_acc, dpi=300)
        # plt.close()
    if is_load == 1:
        model.load_weights(model_filepath).expect_partial()
        # model_PR.load_weights(filepath1)
        # model_PR.evaluate(x_train, y_train, batch_size=25, verbose=1)
        # model_PR.evaluate(x_test, y_test, batch_size=32, verbose=1)
        if output_confusion_matrix == 1:
            y_pred = model.predict(x_test)
            y_pred = y_pred.argmax(axis=1)
            y_test = y_test.argmax(axis=1)
            confusion_mtx = confusion_matrix(y_test, y_pred)
            normalized_confusion_mtx = np.round(confusion_matrix(y_test, y_pred, normalize='true'), 3)
            flatten1 = f1_score(y_true=y_test, y_pred=y_pred, labels=None, average='macro')
            acc = accuracy_score(y_true=y_test, y_pred=y_pred)
            recall = recall_score(y_true=y_test, y_pred=y_pred, labels=None, average='macro')
            preccision = precision_score(y_true=y_test, y_pred=y_pred, labels=None, average='macro')
            # print(confusion_mtx)
            # print(normalized_confusion_mtx)
            print('accuracy: ' + str(acc))
            print('recall: ' + str(recall))
            print('precision: ' + str(preccision))
            print('F1-score: ' + str(flatten1))
            return y_pred


def ELSTM1_api(hidden_units, num_class, prototypes,nu,
                         model_filepath,evi_filepath,mid_filepath,
                         load_and_train, is_load,
                         x_train, y_train, x_test, y_test,
                         output_confusion_matrix):
    inputs = Input((x_train.shape[1], x_train.shape[2]))
    lstm1 = LSTM(units=hidden_units, return_sequences=True, dropout=0, recurrent_dropout=0)(inputs)
    lstm2 = LSTM(units=hidden_units, return_sequences=True, dropout=0, recurrent_dropout=0)(lstm1)
    flatten1 = Flatten()(lstm2)
    # DS layer
    ED = ds_layer.DS1(prototypes, hidden_units)(flatten1)
    ED_ac = ds_layer.DS1_activate(prototypes)(ED)
    mass_prototypes = ds_layer.DS2(prototypes, num_class)(ED_ac)
    mass_prototypes_omega = ds_layer.DS2_omega(prototypes, num_class)(mass_prototypes)
    mass_Dempster = ds_layer.DS3_Dempster(prototypes, num_class)(mass_prototypes_omega)
    mass_Dempster_normalize = ds_layer.DS3_normalize()(mass_Dempster)

    # Utility layer for training
    outputs = utility_layer_train.DM(nu, num_class)(mass_Dempster_normalize)
    model_evi = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model_evi.compile(
        optimizer=keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
                                         schedule_decay=0.004),
        loss='CategoricalCrossentropy',
        metrics=['accuracy'])
    # model_evi.summary()

    if load_and_train == 1:
        # load the weights of probabilistic classifier
        # and get the feature for training DS layers
        model_evi.load_weights(model_filepath).expect_partial()
        feature = tf.keras.Model(inputs=[inputs], outputs=[flatten1])
        train_feature_for_DS = feature.predict(x_train)
        test_feature_for_DS = feature.predict(x_test)

        # training DS layers
        inputss = tf.keras.layers.Input(hidden_units)
        ED = ds_layer.DS1(prototypes, hidden_units)(inputss)
        ED_ac = ds_layer.DS1_activate(prototypes)(ED)
        mass_prototypes = ds_layer.DS2(prototypes, num_class)(ED_ac)
        mass_prototypes_omega = ds_layer.DS2_omega(prototypes, num_class)(mass_prototypes)
        mass_Dempster = ds_layer.DS3_Dempster(prototypes, num_class)(mass_prototypes_omega)
        mass_Dempster_normalize = ds_layer.DS3_normalize()(mass_Dempster)
        outputss = utility_layer_train.DM(nu, num_class)(mass_Dempster_normalize)
        model_mid = tf.keras.Model(inputs=[inputss], outputs=[outputss])
        model_mid.compile(
            optimizer=keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
                                             schedule_decay=0.004),
            # 0.001
            loss='CategoricalCrossentropy',
            metrics=['accuracy'])
        # model_mid.summary()
        mid_callback = ModelCheckpoint(mid_filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                       save_weights_only=True, save_frequency=1)
        h = model_mid.fit(train_feature_for_DS, y_train, batch_size=64, epochs=20, verbose=1,
                          validation_data=(test_feature_for_DS, y_test), callbacks=[mid_callback], shuffle=True)

        # feed the trained paramters to the evidential disusedModel
        model_evi.load_weights(model_filepath).expect_partial()
        DS1_W = tf.reshape(model_mid.layers[1].get_weights()[0], [1, prototypes, hidden_units])
        DS1_activate_W = model_mid.layers[2].get_weights()
        DS2_W = model_mid.layers[3].get_weights()
        model_evi.layers[4].set_weights(DS1_W)
        model_evi.layers[5].set_weights(DS1_activate_W)
        model_evi.layers[6].set_weights(DS2_W)

        checkpoint_callback = ModelCheckpoint(
            evi_filepath, monitor='val_accuracy', verbose=1,
            save_best_only=True, save_weights_only=True,
            save_frequency=1)
        model_evi.fit(x_train, y_train,batch_size=64, epochs=18, verbose=1,
                      callbacks=[checkpoint_callback], validation_data=(x_test, y_test), shuffle=True)

    if is_load == 1:
        model_evi.load_weights(evi_filepath).expect_partial()
        # model_evi.evaluate(x_train, y_train, batch_size=64, verbose=1)
        # model_evi.evaluate(x_test, y_test, batch_size=64, verbose=1)
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
            return y_pred

def ELSTM_set_value_api(hidden_units,num_class, number_act_set, act_set, nu, tol,prototypes,
                  utility_matrix, is_load, filepath,x_test, numerical_y_test):
    inputs = Input((x_test.shape[1], x_test.shape[2]))
    lstm1 = LSTM(units=hidden_units, return_sequences=True, dropout=0, recurrent_dropout=0)(inputs)
    lstm2 = LSTM(units=hidden_units, return_sequences=True, dropout=0, recurrent_dropout=0)(lstm1)
    flatten1 = Flatten()(lstm2)

    # DS layer
    ED = ds_layer.DS1(prototypes, hidden_units)(flatten1)
    ED_ac = ds_layer.DS1_activate(prototypes)(ED)
    mass_prototypes = ds_layer.DS2(prototypes, num_class)(ED_ac)
    mass_prototypes_omega = ds_layer.DS2_omega(prototypes, num_class)(mass_prototypes)
    mass_Dempster = ds_layer.DS3_Dempster(prototypes, num_class)(mass_prototypes_omega)
    mass_Dempster_normalize = ds_layer.DS3_normalize()(mass_Dempster)

    # Utility layer for training
    outputs = utility_layer_test.DM_test(num_class, number_act_set, nu)(mass_Dempster_normalize)
    model_evi_SV = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model_evi_SV.compile(
        optimizer=keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
                                         schedule_decay=0.004), loss='CategoricalCrossentropy', metrics=['accuracy'])
    # model_evi_SV.summary()
    if is_load:
        model_evi_SV.layers[-1].set_weights(tf.reshape(utility_matrix, [1, number_act_set, num_class]))
        model_evi_SV.load_weights(filepath).expect_partial()

        results = tf.argmax(model_evi_SV.predict(x_test), -1)
        # print(model_evi_SV.predict(x_test).shape)
        # print(results.shape)
        imprecise_results = []
        sv_counter = 0
        for i in range(len(results)):
            act_local = results[i]
            set_valued_results = act_set[act_local]
            imprecise_results.append(set_valued_results)

            if len(set_valued_results) > 1:
                sv_counter = sv_counter + 1
        # print(imprecise_results)
        average_utility_imprecision = AU_imprecision.average_utility(utility_matrix, results, numerical_y_test, act_set)
        print('prototypes=' + str(prototypes) + ' nu=' + str(nu) + ' tol=' + str(tol))
        print('**** **** **** **** **** **** AU = ' + str(average_utility_imprecision) + ' **** **** **** **** **** *****')
        print('**** **** **** **** **** **** SV Rates assignment = ' + str((sv_counter / len(results)) * 100) + ' **** **** **** **** **** *****')