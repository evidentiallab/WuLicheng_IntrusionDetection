import numpy as np
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
from keras import layers
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
import model.Set_valued as SV
from scipy.optimize import minimize
import math


def pLSTM(hidden_units, num_class, model_filepath,is_train, is_load,
                     x_train, y_train, x_test, y_test, output_confusion_matrix, pic_filepath_loss=0, pic_filepath_acc=0):

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


def LSTM_SV(hidden_units,data_WIDTH,num_class, number_act_set, act_set, tol, nu,
           utility_matrix, is_load, filepath, x_test, numerical_y_test):
    inputs = Input((x_test.shape[1], x_test.shape[2]))
    lstm1 = LSTM(units=hidden_units, return_sequences=True, dropout=0, recurrent_dropout=0)(inputs)
    lstm2 = LSTM(units=hidden_units, return_sequences=True, dropout=0, recurrent_dropout=0)(lstm1)
    flatten1 = Flatten()(lstm2)
    softmax = Dense(units=num_class, activation='softmax')(flatten1)

    outputs = utility_layer_test.DM_test(num_class, number_act_set, nu)(softmax)
    model_SV = keras.Model(inputs=[inputs], outputs=[outputs])
    model_SV.compile(optimizer=keras.optimizers.Nadam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004),
        loss='CategoricalCrossentropy', metrics=['accuracy'])
    # disusedModel.summary()
    checkpoint_callback = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                          save_weights_only=True, save_frequency=1)
    if is_load:
        model_SV.layers[-1].set_weights(tf.reshape(utility_matrix, [1, number_act_set, num_class]))
        model_SV.load_weights(filepath).expect_partial()

        results = tf.argmax(model_SV.predict(x_test), -1)
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

        average_utility_imprecision = AU_imprecision.average_utility(utility_matrix, results, numerical_y_test, act_set)
        print('**** **** **** **** **** **** tol=' + str(tol) + ' **** **** **** **** **** *****')
        print('**** **** **** **** **** **** AU = ' + str(
            average_utility_imprecision) + ' **** **** **** **** **** *****')
        print('**** **** **** **** **** **** SV Rates assignment = ' + str(
            (sv_counter / len(results)) * 100) + ' **** **** **** **** **** *****')


def Test_LSTM_SV(hidden_units,data_WIDTH,num_class, number_act_set, nu,
                filepath, x_test, numerical_y_test):
    class_set = list(range(num_class))
    act_set = SV.PowerSets(class_set, no_empty=True, is_sorted=True)

    for i in [0, 1, 2, 3, 4]:
        m = 0
        UM = SV.utility_mtx(num_class, act_set=act_set, class_set=class_set, tol_i=i, m=m)
        LSTM_SV(hidden_units=hidden_units, data_WIDTH=data_WIDTH, num_class=num_class, number_act_set=number_act_set, tol=0.1 * i + 0.5 + m,
               utility_matrix=UM, is_load=1, nu=nu,filepath=filepath,
               act_set=act_set, x_test=x_test, numerical_y_test=numerical_y_test)

    for i in [4]:
        m = 0.05
        UM = SV.utility_mtx(num_class, act_set=act_set, class_set=class_set, tol_i=i, m=m)
        LSTM_SV(hidden_units=hidden_units, data_WIDTH=data_WIDTH, num_class=num_class, number_act_set=number_act_set, tol=0.1 * i + 0.5 + m,
               utility_matrix=UM, is_load=1, nu=nu, filepath=filepath,
               act_set=act_set, x_test=x_test, numerical_y_test=numerical_y_test)

