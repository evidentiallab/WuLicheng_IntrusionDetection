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


# aim func: cross entropy
def func(x):
    fun = 0
    for i in range(len(x)):
        fun += x[i] * math.log10(x[i])
    return fun


# constraint 1: the sum of weights is 1
def cons1(x):
    return sum(x)


# constraint 2: define tolerance to imprecision
def cons2(x):
    tol = 0
    for i in range(len(x)):
        tol += (len(x) -(i+1)) * x[i] / (len(x) - 1)
    return tol


# compute the weights g for ordered weighted average aggreagtion
def utility_mtx(num_cls, act_set, class_set):
    num_of_class = num_cls
    # generate and optimize the weights
    for j in range(2, (num_of_class + 1)):
        num_weights = j
        ini_weights = np.asarray(np.random.rand(num_weights))

        name = 'weight' + str(j)
        # 5 rows of weight for imprecision tolerance degree(tol) = 0.5/0.6/0.7/0.8/0.9
        locals()['weight' + str(j)] = np.zeros([5, j])

        for i in range(5):
            tol = 0.5 + i * 0.1

            cons = ({'type': 'eq', 'fun': lambda x: cons1(x) - 1},
                    {'type': 'eq', 'fun': lambda x: cons2(x) - tol},
                    {'type': 'ineq', 'fun': lambda x: x - 0.00000001})

            res = minimize(func, ini_weights, method='SLSQP', options={'disp': True}, constraints=cons)
            # print('res'+str(res))
            locals()['weight' + str(j)][i] = res.x
            print(str(j)+'weight:'+str(res.x))
            # print(locals())

    utility_matrix = np.zeros([len(act_set), len(class_set)])
    # print(utility_matrix.shape)
    tol_i = 4
    # tol_i = 0 with tol=0.5, tol_i = 1 with tol=0.6, tol_i = 2 with tol=0.7, tol_i = 3 with tol=0.8, tol_i = 4 with tol=0.9
    for i in range(len(act_set)):
        intersec = class_set and act_set[i]
        if len(intersec) == 1:
            utility_matrix[i, intersec] = 1
        else:
            for j in range(len(intersec)):
                utility_matrix[i, intersec[j]] = locals()['weight' + str(len(intersec))][tol_i, 0]
    # print(locals()['weight2'])
    # print(utility_matrix.shape)
    return utility_matrix


# power set
def PowerSets(items,no_empty,is_sorted):
    N = len(items)
    set_all = []
    for i in range(2 ** N):
        combo = []
        for j in range(N):
            if (i >> j) % 2 == 1:
                combo.append(items[j])
        set_all.append(combo)
    if is_sorted:
        set_all = sorted(set_all)
    else:
        pass

    if no_empty:
        set_all.remove(set_all[0])
        return set_all
    else:
        return set_all


def set_valued_evidential_FitNet4(num_class, number_act_set, act_set, nu, tol,
                                  prototypes, utility_matrix, is_load, filepath,
                                  x_test, numerical_y_test):
    data_WIDTH = 10
    data_HEIGHT = 10
    inputs_pixels = data_WIDTH * data_HEIGHT
    inputs = tf.keras.layers.Input((data_HEIGHT, data_WIDTH, 1))
    # filepath = 'pickleJar/Evidential/EVI_FitNet_0.001and64_500proto'

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

    # Utility layer for testing
    outputs = utility_layer_test.DM_test(num_class, number_act_set, nu)(mass_Dempster_normalize)
    model_evi_SV = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model_evi_SV.compile(
        optimizer=keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004),
        loss='CategoricalCrossentropy',
        metrics=['accuracy'])
    # model_evi_SV.summary()
    if is_load:
        model_evi_SV.layers[-1].set_weights(tf.reshape(utility_matrix, [1, 31, 5]))
        model_evi_SV.load_weights(filepath).expect_partial()

        results = tf.argmax(model_evi_SV.predict(x_test), -1)
        # print(model_evi_SV.predict(x_test).shape)
        # print(results.shape)
        imprecise_results = []
        for i in range(len(results)):
            act_local = results[i]
            set_valued_results = act_set[act_local]
            imprecise_results.append(set_valued_results)
        print(imprecise_results)
        average_utility_imprecision = AU_imprecision.average_utility(utility_matrix, results, numerical_y_test, act_set)
        print('prototypes='+str(prototypes)+' nu='+str(nu)+' tol='+str(tol))
        print('AU = ' + str(average_utility_imprecision))


if __name__ == '__main__':
    pass
    # num_class = 5
    #
    # class_set = list(range(num_class))
    # act_set = PowerSets(class_set,no_empty=True,is_sorted=True)
    # print(len(act_set))
    # UM = utility_mtx(num_class, actset=act_set, class_set=class_set)
    #
    # # label_dict = {0: 'normal', 1: 'Dos', 2: 'Probe', 3: 'U2R', 4: 'R2L'}
    # # label_powerset = []
    # # for sublist in act_set:
    # #     label_set = []
    # #     for num in sublist:
    # #         label_set.append(label_dict[num])
    # #     label_powerset.append(label_set)
    # #
    # # print(label_powerset)
    # # print(len(label_powerset))
    #
    # # read the training set
    # trainDF = pd.read_csv('../dataset/KDDCUP99/Encoded/train(123_100).csv', header=None)
    # trainND = trainDF.values
    # train2D = trainND.reshape(trainND.shape[0], 10, 10)
    # train_labelDF = pd.read_csv('../dataset/KDDCUP99/Label/trainLabel/multi-label-onehot.csv', header=0)
    # train_label = train_labelDF.values
    # # read the test set
    # testDF = pd.read_csv('../dataset/KDDCUP99/Encoded/test(123_100).csv', header=None)
    # testND = testDF.values
    # test2D = testND.reshape(testND.shape[0], 10, 10)
    # test_labelDF = pd.read_csv('../dataset/KDDCUP99/Label/testLabel/multi-label-onehot.csv', header=0)
    # test_label = test_labelDF.values
    #
    # # prototypes=200/400/500
    # set_valued_evidential_FitNet4(num_class=5,number_act_set=31,
    #                               prototypes=500,utility_matrix=UM,is_load=True,act_set=act_set,
    #                               x_train=train2D, y_train=train_label,
    #                               x_test=test2D, y_test=test_label)

