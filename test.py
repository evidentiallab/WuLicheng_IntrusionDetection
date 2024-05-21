import numpy as np
import random
import tensorflow as tf
import pandas as pd
from disusedModel import ONEDCNN_NB15
import model
import model.ECNN_UNSW_NB15 as ECNN_UNSW_NB15
import model.ECNN_NSL_KDD as ECNN_NSL_KDD
import model.Set_valued as sv
import model.ERNN as ERNN
import model.ELSTM as ELSTM

from scipy.stats import chi2_contingency
from sklearn.model_selection import KFold, StratifiedKFold
import model.Set_valued as SV
import os


def load_UNSWNB15(split, is_reshape):
    if split == "official":
        X_train = pd.read_csv('D:/NB15dataset/reselect_X_train0.csv', header=0).values
        X_test = pd.read_csv('D:/NB15dataset/reselect_X_test0.csv', header=0).values

        y = pd.read_csv('D:/NB15dataset/concat_label_onehot.csv', header=0)
        y_train = y.iloc[0:175341].values
        y_test = y.iloc[175341:].values
        numerical_y_test = [np.argmax(i) for i in y_test]
        print("official split")

    elif split == "random":
        X_train = pd.read_csv('D:/NB15dataset/reselect_X_train1.csv', header=0)
        X_test = pd.read_csv('D:/NB15dataset/reselect_X_test1.csv', header=0)

        y_train = pd.read_csv('D:/NB15dataset/reselect_y_train1.csv', header=0).values
        y_test = pd.read_csv('D:/NB15dataset/reselect_y_test1.csv', header=0).values
        numerical_y_test = [np.argmax(i) for i in y_test]
        print("random split")

    if is_reshape == True:
        x_train = pd.read_csv('D:/NB15dataset/reselect_X_train1.csv', header=0).values
        x_test = pd.read_csv('D:/NB15dataset/reselect_X_test1.csv', header=0).values
        x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

        X_train = pd.DataFrame(x_train)
        X_test = pd.DataFrame(x_test)

    return X_train, X_test, y_train, y_test, numerical_y_test


def check_Model_in_Selective_SV(num_class, number_act_set, selective_set_UNSWNB15,
                filepath, x_test, numerical_y_test):
    class_set = list(range(num_class))
    act_set = SV.PowerSets(class_set, no_empty=True, is_sorted=True)
    # print(act_set)
    # selective_set_NSLKDD = [[0],[0, 3],[1],[2],[3],[4]]
    # selective_set_UNSWNB15 = [[0], [0,1,2,3], [0, 1, 2, 3, 9], [1], [1,2], [1,2,3], [2], [3], [4], [5], [6], [7], [8], [9]]
    AU_list = []
    SVR_list = []
    for i in [0, 1, 2, 3, 4]:
        m = 0
        UM = SV.utility_mtx(num_class, act_set=selective_set_UNSWNB15, class_set=class_set, tol_i=i, m=m)
        AU, SVR = ELSTM.LSTM_SV(hidden_units=60,num_class=num_class, number_act_set=number_act_set, act_set=selective_set_UNSWNB15,
                                tol=0.1 * i + 0.5 + m, nu=0.5, utility_matrix=UM, is_load=True,
                                filepath=filepath,x_test=x_test, numerical_y_test=numerical_y_test)
        AU_list.append(AU)
        SVR_list.append(SVR)

    for i in [4]:
        m = 0.099
        UM = SV.utility_mtx(num_class, act_set=selective_set_UNSWNB15, class_set=class_set, tol_i=i, m=m)
        AU, SVR = ELSTM.LSTM_SV(hidden_units=60,num_class=num_class, number_act_set=number_act_set, act_set=selective_set_UNSWNB15,
                                tol=0.1 * i + 0.5 + m, nu=0.5,utility_matrix=UM, is_load=True,
                                filepath=filepath,x_test=x_test, numerical_y_test=numerical_y_test)
        AU_list.append(AU)
        SVR_list.append(SVR)
    return AU_list, SVR_list


if __name__ == '__main__':
    X, X_test, y, y_test, numerical_y_test = load_UNSWNB15(split="random", is_reshape=False)
    numerical_y = [np.argmax(i) for i in y]
    numerical_y = pd.DataFrame(numerical_y)
    onehot_y = pd.read_csv('D:/NB15dataset/reselect_y_train1.csv', header=0)

    AU_record = []
    SVR_record = []
    selective_set_1 = [[0], [1], [1,2], [2], [3], [4], [5], [6], [7], [8], [9]]
    selective_set_2 = [[0], [1], [1,2], [1,2,3], [2], [3], [4], [5], [6], [7], [8], [9]]
    selective_set_3 = [[0], [0,1,2,3], [1], [1,2], [1,2,3], [2], [3], [4], [5], [6], [7], [8], [9]]
    selective_set_4 = [[0], [0,1,2,3], [0,1,2,3,9], [1], [1,2], [1,2,3], [2], [3], [4], [5], [6], [7], [8], [9]]
    selective_set_5 = [[0], [0,1,2,3], [0,1,2,3,9], [1], [1,2], [1,2,3], [2], [3], [4], [4,8], [5], [6], [7], [8], [9]]
    # for selective_set in [selective_set_1,selective_set_2,selective_set_3,selective_set_4,selective_set_5]:
    #     print("*********************************集值标签数量=" + str(len(selective_set)) + "*********************************")
    for counter in [1, 2, 3, 4]:
        X_np, X_test_np = X.values, X_test.values
        X_reshape, X_test_reshape = X_np.reshape((X_np.shape[0], 1, X_np.shape[1])), X_test_np.reshape((X_test_np.shape[0], 1, X_test_np.shape[1]))
        AU, SVR = check_Model_in_Selective_SV(num_class=10, number_act_set=len(selective_set_4), selective_set_UNSWNB15=selective_set_4,
                                            filepath='pickleJarNB15/KFOLD/LSTM/dense64_random_KFOLD%d' % counter,
                                            x_test=X_test_reshape, numerical_y_test=numerical_y_test)
        AU_record.append(AU)
        SVR_record.append(SVR)
    AU_np = np.array(AU_record)
    SVR_np = np.array(SVR_record)
    AU_mean = np.mean(AU_np, axis=0)
    SVR_np = np.mean(SVR_np, axis=0)
    print("*********************************集值标签数量=" + str(len(selective_set_4)) + "*********************************")
    print("AU mean:" + str(AU_mean))
    print("SV Rate mean:" + str(SVR_np))
    print("*********************************集值标签数量=" + str(len(selective_set_4)) + "*********************************")