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
from keras.layers import Dense, LSTM, MaxPool1D, Flatten, Dropout
from keras.models import Sequential
from keras.layers import Input
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint
from libs import ds_layer               # Dempster-Shafer layer
from libs import utility_layer_train    # Utility layer for training
from libs import utility_layer_test     # Utility layer for testing
from libs import AU_imprecision         # Metric average utility for set-valued classification


def probabilistic_LSTM(hidden_units,num_class,model_filepath,lr,
         is_train, is_load,x_train,y_train,x_test,y_test,pic_filepath_loss,pic_filepath_acc,output_confusion_matrix):

    model = Sequential()
    model.add(LSTM(hidden_units, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(hidden_units, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(hidden_units, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(units=50))
    model.add(Dense(units=num_class, activation='softmax'))

    # model.compile(optimizer=keras.optimizers.Nadam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=None,
    #                                                schedule_decay=0.004), loss='CategoricalCrossentropy',metrics=['accuracy'])

    model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr, momentum=0.0, nesterov=False),
                  loss='CategoricalCrossentropy',metrics=['accuracy'])
    # model.summary()

    checkpoint_callback = ModelCheckpoint(model_filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                          save_weights_only=True, save_frequency=1)
    if is_train == 1:
        h = model.fit(x_train, y_train, batch_size=32, epochs=60,
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
            # print(confusion_mtx)
            # print(normalized_confusion_mtx)

            print('accuracy: ' + str(acc))
            print('recall: ' + str(recall))
            print('precision: ' + str(preccision))
            print('F1-score: ' + str(f1))


def bilayer_LSTM(hidden_units,num_class,model_filepath,lr,
         is_train, is_load,x_train,y_train,x_test,y_test,pic_filepath_loss,pic_filepath_acc,output_confusion_matrix):

    model = Sequential()
    model.add(LSTM(hidden_units, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(hidden_units, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(units=50))
    model.add(Dense(units=num_class, activation='softmax'))

    # model.compile(optimizer=keras.optimizers.Nadam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=None,
    #                                                schedule_decay=0.004), loss='CategoricalCrossentropy',metrics=['accuracy'])

    # model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr, momentum=0.0, nesterov=False),
    #               loss='CategoricalCrossentropy',metrics=['accuracy'])
    model.compile(optimizer='adam',loss='CategoricalCrossentropy', metrics=['accuracy'])
    # model.summary()

    checkpoint_callback = ModelCheckpoint(model_filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                          save_weights_only=True, save_frequency=1)
    if is_train == 1:
        h = model.fit(x_train, y_train, batch_size=32, epochs=60,
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
            # print(confusion_mtx)
            # print(normalized_confusion_mtx)

            print('accuracy: ' + str(acc))
            print('recall: ' + str(recall))
            print('precision: ' + str(preccision))
            print('F1-score: ' + str(f1))


def singlelayer_LSTM_sequential(hidden_units, num_class, model_filepath, lr,
                                is_train, is_load, x_train, y_train, x_test, y_test, pic_filepath_loss, pic_filepath_acc, output_confusion_matrix):

    model = Sequential()
    model.add(LSTM(hidden_units, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(units=50))
    model.add(Dense(units=num_class, activation='softmax'))

    # model.compile(optimizer=keras.optimizers.Nadam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=None,
    #                                                schedule_decay=0.004), loss='CategoricalCrossentropy',metrics=['accuracy'])

    # model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr, momentum=0.0, nesterov=False),
    #               loss='CategoricalCrossentropy',metrics=['accuracy'])
    model.compile(optimizer='adam',loss='CategoricalCrossentropy', metrics=['accuracy'])
    # model.summary()

    checkpoint_callback = ModelCheckpoint(model_filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                          save_weights_only=True, save_frequency=1)
    if is_train == 1:
        h = model.fit(x_train, y_train, batch_size=32, epochs=80,
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
            # print(confusion_mtx)
            # print(normalized_confusion_mtx)
            print('accuracy: ' + str(acc))
            print('recall: ' + str(recall))
            print('precision: ' + str(preccision))
            print('F1-score: ' + str(f1))


def singlelayer_LSTM_api(hidden_units, num_class, model_filepath, lr,
                         is_train, is_load, x_train, y_train, x_test, y_test, pic_filepath_loss, pic_filepath_acc, output_confusion_matrix):

    inputs = keras.layers.Input((x_train.shape[1], x_train.shape[2]))
    lstm1 = keras.layers.LSTM(units=hidden_units,return_sequences=True,dropout=0.2)(inputs)
    flatten1 = keras.layers.Flatten()(lstm1)
    dense1 = keras.layers.Dense(units=80)(flatten1)
    outputs = keras.layers.Dense(units=num_class, activation='softmax')(dense1)
    model = keras.Model(inputs=[inputs], outputs=[outputs])

    # model.compile(optimizer=keras.optimizers.Nadam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=None,
    #                                                schedule_decay=0.004), loss='CategoricalCrossentropy',metrics=['accuracy'])

    # model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr, momentum=0.0, nesterov=False),
    #               loss='CategoricalCrossentropy',metrics=['accuracy'])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-08),
                  loss='CategoricalCrossentropy', metrics=['accuracy'])
    # model.summary()

    checkpoint_callback = ModelCheckpoint(model_filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                          save_weights_only=True, save_frequency=1)
    if is_train == 1:
        h = model.fit(x_train, y_train, batch_size=32, epochs=50,
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
            # print(y_pred)
            # print(y_test)
            # print(type(y_pred))
            # print(type(y_test))
            # print(y_test.shape)
            # print(y_pred.shape)
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


def evidential_LSTM_API(hidden_units, num_class, model_filepath, nu, load_weights, train_DS_Layers, prototypes, evi_filepath, mid_filepath,
                        is_load, x_train, y_train, x_test, y_test, output_confusion_matrix):
    inputs = keras.layers.Input((x_train.shape[1], x_train.shape[2]))
    lstm1 = keras.layers.LSTM(units=hidden_units, return_sequences=True, dropout=0.2)(inputs)
    flatten1 = keras.layers.Flatten()(lstm1)
    dense1 = keras.layers.Dense(units=80)(flatten1)

    # DS layer
    ED = ds_layer.DS1(prototypes, 80)(dense1)
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

    if load_weights == 1:
        # load the weights of probabilistic classifier
        # and get the feature for training DS layers
        model_evi.load_weights(model_filepath).expect_partial()
        feature = tf.keras.Model(inputs=[inputs], outputs=[flatten1])
        train_feature_for_DS = feature.predict(x_train)
        test_feature_for_DS = feature.predict(x_test)

    if train_DS_Layers == 1:
        # training DS layers
        inputss = tf.keras.layers.Input(80)
        ED = ds_layer.DS1(prototypes, 80)(inputss)
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
        h = model_mid.fit(train_feature_for_DS, y_train, batch_size=64, epochs=10, verbose=1,
                          validation_data=(test_feature_for_DS, y_test), callbacks=[mid_callback], shuffle=True)

        # feed the trained paramters to the evidential model
        model_evi.load_weights(model_filepath).expect_partial()
        DS1_W = tf.reshape(model_mid.layers[1].get_weights()[0], [1, prototypes, 80])
        DS1_activate_W = model_mid.layers[2].get_weights()
        DS2_W = model_mid.layers[3].get_weights()
        model_evi.layers[4].set_weights(DS1_W)
        model_evi.layers[5].set_weights(DS1_activate_W)
        model_evi.layers[6].set_weights(DS2_W)

        checkpoint_callback = ModelCheckpoint(
            evi_filepath, monitor='val_accuracy', verbose=1,
            save_best_only=True, save_weights_only=True,
            save_frequency=1)
        model_evi.fit(x_train, y_train,
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


def evidential_LSTM_sequential(hidden_units, num_class, nu, is_train, prototypes, evi_filepath,
                        is_load, x_train, y_train, x_test, y_test, output_confusion_matrix):
    model = Sequential()
    model.add(LSTM(hidden_units, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(units=50))
    # model.add(Dense(units=num_class, activation='softmax'))

    # DS layer
    model.add(ds_layer.DS1(prototypes,50))
    model.add(ds_layer.DS1_activate(prototypes))
    model.add(ds_layer.DS2(prototypes, num_class))
    model.add(ds_layer.DS2_omega(prototypes, num_class))
    model.add(ds_layer.DS3_Dempster(prototypes, num_class))
    model.add(ds_layer.DS3_normalize())
    # Utility layer for training
    model.add(utility_layer_train.DM(nu,num_class))

    # model.compile(optimizer=keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004),loss='CategoricalCrossentropy',metrics=['accuracy'])
    model.compile(optimizer='adam', loss='CategoricalCrossentropy',metrics=['accuracy'])
    # model.summary()
    checkpoint_callback = ModelCheckpoint(evi_filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                          save_weights_only=True, save_frequency=1)
    if is_train == 1:
        h = model.fit(x_train, y_train, batch_size=64, epochs=60,
                 verbose=1, callbacks=[checkpoint_callback], validation_data=(x_test, y_test), shuffle=True)
        # history = h.history
        # epochs = range(len(history['loss']))
        # plt.plot(epochs, history['loss'], 'royalblue', label='Train loss')
        # plt.plot(epochs, history['val_loss'], 'crimson', label='Valid loss')  # Test loss
        # plt.xlabel("Epochs")
        # plt.ylabel("Loss")
        # plt.rcParams["figure.dpi"] = 300
        # plt.legend()
        # plt.savefig(pic_filepath_loss, dpi=300)
        # # plt.show()
        # plt.close()
        #
        # plt.plot(epochs, history['accuracy'], 'royalblue', label='Train accuracy')
        # plt.plot(epochs, history['val_accuracy'], 'crimson', label='Valid accuracy')  # Test loss
        # plt.xlabel("Epochs")
        # plt.ylabel("Accuracy")
        # plt.rcParams["figure.dpi"] = 300
        # plt.legend()
        # plt.savefig(pic_filepath_acc, dpi=300)
        # # plt.show()
        # plt.close()
    if is_load == 1:
        model.load_weights(evi_filepath).expect_partial()
        y_pred = model.predict(x_test)
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




