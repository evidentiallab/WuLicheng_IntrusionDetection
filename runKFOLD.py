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


def load_NSLKDD(split, is_reshape):
    if split == "official":
        X_train = pd.read_csv('D:/IDdataset/X_train0.csv', header=0).values
        X_test = pd.read_csv('D:/IDdataset/X_test0.csv', header=0).values

        y = pd.read_csv('D:/IDdataset/reprocess-524/concat_onehot_ordered.csv', header=0)
        y = y.iloc[:, 129:134]
        y_train = y.iloc[0:125973].values
        y_test = y.iloc[125973:].values
        numerical_y_test = [np.argmax(i) for i in y_test]
        print("official split")

    elif split == "random":
        X_train = pd.read_csv('D:/IDdataset/X_train1.csv', header=0)
        X_test = pd.read_csv('D:/IDdataset/X_test1.csv', header=0)

        y_train = pd.read_csv('D:/IDdataset/y_train1.csv', header=0).values
        y_test = pd.read_csv('D:/IDdataset/y_test1.csv', header=0).values
        numerical_y_test = [np.argmax(i) for i in y_test]
        print("random split")

    if is_reshape == True:
        x_train = pd.read_csv('D:/IDdataset/X_train1.csv', header=0).values
        x_test = pd.read_csv('D:/IDdataset/X_test1.csv', header=0).values
        x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

        X_train = pd.DataFrame(x_train)
        X_test = pd.DataFrame(x_test)
    return X_train, X_test, y_train, y_test, numerical_y_test


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


def check_Model_in_SV(num_class, number_act_set, proto,
                filepath, x_test, numerical_y_test):
    class_set = list(range(num_class))
    act_set = SV.PowerSets(class_set, no_empty=True, is_sorted=True)
    AU_list = []
    SVR_list = []
    for i in [0, 1, 2, 3, 4]:
        m = 0
        UM = SV.utility_mtx(num_class, act_set=act_set, class_set=class_set, tol_i=i, m=m)
        # AU, SVR = ECNN_NSL_KDD.ECNN_SV(data_WIDTH, num_class, number_act_set, act_set,
        #                                nu=0.5, tol=0.1 * i + 0.5 + m, prototypes=proto,utility_matrix=UM,
        #                                is_load=True, filepath=filepath, x_test=x_test, numerical_y_test=numerical_y_test)
        AU, SVR = ECNN_NSL_KDD.ECNN_SV(data_WIDTH=40,num_class=num_class, number_act_set=number_act_set, act_set=act_set, prototypes=proto, tol=0.1 * i + 0.5 + m, nu=0.5,
                                      utility_matrix=UM, is_load=True, filepath=filepath, x_test=x_test, numerical_y_test=numerical_y_test)
        # AU, SVR = ELSTM.ELSTM_SV(num_class=num_class, number_act_set=number_act_set, act_set=act_set,hidden_units=60,
        #                          nu=0.5, tol=0.1 * i + 0.5 + m, prototypes=proto,utility_matrix=UM, is_load=True,
        #                          filepath=filepath, x_test=x_test, numerical_y_test=numerical_y_test)
        # AU, SVR = ELSTM.LSTM_SV(hidden_units=60,num_class=num_class, number_act_set=number_act_set, act_set=act_set,
        #                         tol=0.1 * i + 0.5 + m, nu=0.5,utility_matrix=UM, is_load=True,
        #                         filepath=filepath,x_test=x_test, numerical_y_test=numerical_y_test)
        # AU, SVR = ERNN.RNN_SV(hidden_units=64, num_class=num_class, number_act_set=number_act_set, act_set=act_set, tol=0.1 * i + 0.5 + m, nu=0.5,
        #    utility_matrix=UM, is_load=True, filepath=filepath, x_test=x_test, numerical_y_test=numerical_y_test)
        # AU, SVR = ERNN.ERNN_SV(hidden_units=64,num_class=num_class, number_act_set=number_act_set, act_set=act_set, tol=0.1 * i + 0.5 + m, nu=0.5,
        #                        prototypes=proto,utility_matrix=UM, is_load=True, filepath=filepath, x_test=x_test, numerical_y_test=numerical_y_test)
        AU_list.append(AU)
        SVR_list.append(SVR)

    for i in [4]:
        m = 0.05
        UM = SV.utility_mtx(num_class, act_set=act_set, class_set=class_set, tol_i=i, m=m)
        # AU, SVR = ECNN_NSL_KDD.ECNN_SV(data_WIDTH, num_class, number_act_set, act_set,
        #                                nu=0.5, tol=0.1 * i + 0.5 + m,prototypes=proto,utility_matrix=UM,
        #                                is_load=True, filepath=filepath, x_test=x_test, numerical_y_test=numerical_y_test)
        # AU, SVR = ECNN_NSL_KDD.CNN_SV(data_WIDTH, num_class, number_act_set, act_set, tol=0.1 * i + 0.5 + m, nu=0.5,
        #                               utility_matrix=UM, is_load=True, filepath=filepath, x_test=x_test,
        #                               numerical_y_test=numerical_y_test)
        # AU, SVR = ELSTM.ELSTM_SV(num_class=num_class, number_act_set=number_act_set, act_set=act_set,hidden_units=60, nu=0.5, tol=0.1 * i + 0.5 + m,
        #                          prototypes=proto, utility_matrix=UM, is_load=True,
        #                          filepath=filepath, x_test=x_test, numerical_y_test=numerical_y_test)
        # AU, SVR = ELSTM.LSTM_SV(hidden_units=60, num_class=num_class, number_act_set=number_act_set, act_set=act_set,
        #                         tol=0.1 * i + 0.5 + m, nu=0.5, utility_matrix=UM, is_load=True,
        #                         filepath=filepath, x_test=x_test, numerical_y_test=numerical_y_test)
        # AU, SVR = ERNN.RNN_SV(hidden_units=64, num_class=num_class, number_act_set=number_act_set, act_set=act_set, tol=0.1 * i + 0.5 + m, nu=0.5,
        #    utility_matrix=UM, is_load=True, filepath=filepath, x_test=x_test, numerical_y_test=numerical_y_test)
        AU, SVR = ECNN_NSL_KDD.ECNN_SV(data_WIDTH=40, num_class=num_class, number_act_set=number_act_set, prototypes=proto,
                                      act_set=act_set, tol=0.1 * i + 0.5 + m, nu=0.5,
                                      utility_matrix=UM, is_load=True, filepath=filepath, x_test=x_test,
                                      numerical_y_test=numerical_y_test)
        AU_list.append(AU)
        SVR_list.append(SVR)
    return AU_list, SVR_list


def check_Model_in_Selective_SV(num_class, number_act_set, proto,
                filepath, x_test, numerical_y_test):
    class_set = list(range(num_class))
    act_set = SV.PowerSets(class_set, no_empty=True, is_sorted=True)
    selective_set_NSLKDD = [[0],[0, 3],[1],[2],[3],[4]]
    # print(class_set)
    # print
    AU_list = []
    SVR_list = []
    for i in [0, 1, 2, 3, 4]:
        m = 0
        UM = SV.utility_mtx(num_class, act_set=selective_set_NSLKDD, class_set=class_set, tol_i=i, m=m)
    #     # AU, SVR = ECNN_NSL_KDD.ECNN_SV(data_WIDTH=40, num_class=num_class, number_act_set=number_act_set, act_set=selective_set_NSLKDD, prototypes=proto, tol=0.1 * i + 0.5 + m, nu=0.5,
    #     #                               utility_matrix=UM, is_load=True, filepath=filepath, x_test=x_test, numerical_y_test=numerical_y_test)
    #     # AU, SVR = ELSTM.LSTM_SV(hidden_units=60, num_class=num_class, number_act_set=number_act_set, act_set=selective_set_NSLKDD,
    #     #                         tol=0.1 * i + 0.5 + m, nu=0.5, utility_matrix=UM, is_load=True,
    #     #                         filepath=filepath, x_test=x_test, numerical_y_test=numerical_y_test)
    #     # AU, SVR = ERNN.RNN_SV(hidden_units=64, num_class=num_class, number_act_set=number_act_set, act_set=selective_set_NSLKDD, tol=0.1 * i + 0.5 + m, nu=0.5,
    #     #    utility_matrix=UM, is_load=True, filepath=filepath, x_test=x_test, numerical_y_test=numerical_y_test)
    #     # AU, SVR = ERNN.ERNN_SV(hidden_units=64,num_class=num_class, number_act_set=number_act_set, act_set=selective_set_NSLKDD, tol=0.1 * i + 0.5 + m, nu=0.5,
    #     #                        prototypes=proto,utility_matrix=UM, is_load=True, filepath=filepath, x_test=x_test, numerical_y_test=numerical_y_test)
    #     AU, SVR = ELSTM.ELSTM_SV(num_class=num_class, number_act_set=number_act_set, act_set=selective_set_NSLKDD, hidden_units=60,
    #                              nu=0.5, tol=0.1 * i + 0.5 + m, prototypes=proto, utility_matrix=UM, is_load=True,
    #                              filepath=filepath, x_test=x_test, numerical_y_test=numerical_y_test)
    #     AU_list.append(AU)
    #     SVR_list.append(SVR)

    for i in [4]:
        m = 0.099
        UM = SV.utility_mtx(num_class, act_set=selective_set_NSLKDD, class_set=class_set, tol_i=i, m=m)
        # AU, SVR = ECNN_NSL_KDD.ECNN_SV(data_WIDTH=40,num_class=num_class, number_act_set=number_act_set, act_set=selective_set_NSLKDD, prototypes=proto, tol=0.1 * i + 0.5 + m, nu=0.5,
        #                               utility_matrix=UM, is_load=True, filepath=filepath, x_test=x_test, numerical_y_test=numerical_y_test)
        # AU, SVR = ELSTM.LSTM_SV(hidden_units=60, num_class=num_class, number_act_set=number_act_set, act_set=selective_set_NSLKDD,
        #                         tol=0.1 * i + 0.5 + m, nu=0.5, utility_matrix=UM, is_load=True,
        #                         filepath=filepath, x_test=x_test, numerical_y_test=numerical_y_test)
        # AU, SVR = ERNN.RNN_SV(hidden_units=64, num_class=num_class, number_act_set=number_act_set, act_set=selective_set_NSLKDD, tol=0.1 * i + 0.5 + m, nu=0.5,
        #    utility_matrix=UM, is_load=True, filepath=filepath, x_test=x_test, numerical_y_test=numerical_y_test)
        # AU, SVR = ERNN.ERNN_SV(hidden_units=64, num_class=num_class, number_act_set=number_act_set,
        #                        act_set=selective_set_NSLKDD, tol=0.1 * i + 0.5 + m, nu=0.5,
        #                        prototypes=proto, utility_matrix=UM, is_load=True, filepath=filepath, x_test=x_test,
        #                        numerical_y_test=numerical_y_test)
        AU, SVR = ELSTM.ELSTM_SV(num_class=num_class, number_act_set=number_act_set, act_set=selective_set_NSLKDD, hidden_units=60,
                                 nu=0.5, tol=0.1 * i + 0.5 + m, prototypes=proto, utility_matrix=UM, is_load=True,
                                 filepath=filepath, x_test=x_test, numerical_y_test=numerical_y_test)
        AU_list.append(AU)
        SVR_list.append(SVR)
    return AU_list, SVR_list


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    X, X_test, y, y_test, numerical_y_test = load_NSLKDD(split="random", is_reshape=False)
    numerical_y = [np.argmax(i) for i in y]
    numerical_y = pd.DataFrame(numerical_y)
    onehot_y = pd.read_csv('D:/IDdataset/y_train1.csv', header=0)
    # counter = 0
    # kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    # for train_index, val_index in kfold.split(X, numerical_y):
    #     counter = counter + 1
    #     X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    #     y_train, y_val = onehot_y.iloc[train_index], onehot_y.iloc[val_index]
    #
    #     """ ================================================================================================="""
    #     x_train, x_val = X_train.values, X_val.values
    #     X_train, X_val = x_train.reshape((x_train.shape[0], 1, x_train.shape[1])), x_val.reshape((x_val.shape[0], 1, x_val.shape[1]))
    #
    #     ERNN.ERNN_direct_training(hidden_units=64, num_class=5, prototypes=64, nu=0.5, is_train=True,
    #                             evi_filepath='pickleJarNSL/KFOLD/ERNN/proto64_dense32_random_KFOLD%d' % counter,
    #                             x_train=X_train, y_train=y_train, x_test=X_val, y_test=y_val,output_confusion_matrix=1)
    #     ERNN.RNN(hidden_units=64, num_class=5, model_filepath='pickleJarNSL/KFOLD/RNN/dense32_random_KFOLD%d' % counter,
    #              is_train=True, x_train=X_train, y_train=y_train, x_test=X_val, y_test=y_val, output_confusion_matrix=1)
    #
    #     ELSTM.pLSTM(hidden_units=60, num_class=5, is_train=True,
    #                 model_filepath='pickleJarNSL/KFOLD/LSTM/dense64_random_KFOLD%d' % counter,
    #                 x_train=X_train, y_train=y_train, x_test=X_val, y_test=y_val,output_confusion_matrix=1)
    #     ELSTM.ELSTM_direct_training(hidden_units=60, num_class=5, prototypes=70, nu=0.5, is_train=False,
    #                                 evi_filepath='pickleJarNSL/KFOLD/ELSTM/dense64_noalpha_random_KFOLD%d' % counter,
    #                                 x_train=X_train, y_train=y_train, x_test=X_val, y_test=y_val, output_confusion_matrix=1)
    #     """ ================================================================================================="""
    #
    #     ECNN_NSL_KDD.CNN(data_WIDTH=40, num_class=5,is_train=True,
    #                      model_filepath='pickleJarNSL/KFOLD/CNN/random_KFOLD%d' % counter,
    #                      x_train=X_train, y_train=y_train, x_test=X_val, y_test=y_val,output_confusion_matrix=1)
    #     ECNN_NSL_KDD.ECNN_direct_training(data_WIDTH=40, num_class=5, prototypes=72, nu=0.5, flatten_size=64, is_train=True,
    #                                       evi_filepath='pickleJarNSL/KFOLD/ECNN_direct/noalpha_random_proto72_nu0.5_KFOLD%d' % counter,
    #                                       epochs=90,  x_train=X_train, y_train=y_train, x_test=X_val, y_test=y_val,
    #                                       output_confusion_matrix=1)
    #     ECNN_NSL_KDD.ERBFNN(data_WIDTH=40, num_class=5, prototypes=72, nu=0.5, is_train=True,
    #                         epochs=90, model_filepath='pickleJarNSL/KFOLD/ERBFNN/noalpha_random_proto72_nu0.5_KFOLD%d' % counter,
    #                         x_train=X_train, y_train=y_train, x_test=X_val, y_test=y_val,
    #                         output_confusion_matrix=1)
    #
    #     ECNN_UNSW_NB15.ECNN_direct_training(data_WIDTH=50, num_class=10, prototypes=40, nu=0.5, flatten_size=64, is_train=True,
    #                                       evi_filepath='pickleJarNB15/KFOLD/ECNN_direct/random_proto40_nu0.5_KFOLD%d' % counter,
    #                                       epochs=90,  x_train=X_train, y_train=y_train, x_test=X_val, y_test=y_val,output_confusion_matrix=1)
    #     ECNN_UNSW_NB15.ERBFNN(data_WIDTH=50, num_class=10, prototypes=40, nu=0.5, is_train=True,
    #                           epochs=90, model_filepath='pickleJarNB15/KFOLD/ERBFNN/random_proto40_nu0.5_KFOLD%d' % counter,
    #                           x_train=X_train, y_train=y_train, x_test=X_val, y_test=y_val,output_confusion_matrix=1)

    # a1 = r1 = p1 = f1 = 0
    AU_record = []
    SVR_record = []
    for counter in [1,2,3,4]:
        X_np, X_test_np = X.values, X_test.values
        X_reshape, X_test_reshape = X_np.reshape((X_np.shape[0], 1, X_np.shape[1])), X_test_np.reshape((X_test_np.shape[0], 1, X_test_np.shape[1]))
    #     acc1, recall1, prec1, fs1 = ELSTM.pLSTM(hidden_units=60, num_class=5, is_train=False,
    #                                 model_filepath='pickleJarNSL/KFOLD/LSTM/dense64_random_KFOLD%d' % counter,
    #                                 x_train=X_reshape, y_train=y, x_test=X_reshape, y_test=y, output_confusion_matrix=1)
    #     a1 = a1 + acc1; r1 = r1 + recall1; p1 = p1 + prec1; f1 = f1 + fs1
    # print('Performance on NSL-KDD')
    # print("accuracy: %f   ; recall: %f   ; precision: %f   ; f1_score: %f" % (a1 / 4, r1 / 4, p1 / 4, f1 / 4))
    #     AU, SVR = check_Model_in_SV(num_class=5, number_act_set=31, proto=70,
    #                       filepath="pickleJarNSL/KFOLD/CNN/random_KFOLD%d" % counter,
    #                       x_test=X_test_np, numerical_y_test=numerical_y_test)
    #     AU, SVR = check_Model_in_Selective_SV(num_class=5, number_act_set=6, proto=72,
    #                       filepath="pickleJarNSL/KFOLD/ECNN_direct/noalpha_random_proto72_nu0.5_KFOLD%d" % counter,
    #                       x_test=X_test_np, numerical_y_test=numerical_y_test)
        AU, SVR = check_Model_in_Selective_SV(num_class=5, number_act_set=6, proto=70,
                                              filepath="pickleJarNSL/KFOLD/ELSTM/dense64_random_KFOLD%d" % counter,
                                              x_test=X_test_reshape, numerical_y_test=numerical_y_test)
        AU_record.append(AU)
        SVR_record.append(SVR)
    AU_np = np.array(AU_record)
    SVR_np = np.array(SVR_record)
    AU_mean = np.mean(AU_np, axis=0)
    SVR_np = np.mean(SVR_np, axis=0)
    print("AU mean:" + str(AU_mean))
    print("SV Rate mean:" + str(SVR_np))
    #
    #


    X, X_test, y, y_test, numerical_y_test = load_UNSWNB15(split="random", is_reshape=False)
    numerical_y = [np.argmax(i) for i in y]
    numerical_y = pd.DataFrame(numerical_y)
    onehot_y = pd.read_csv('D:/NB15dataset/reselect_y_train1.csv', header=0)
    # counter = 0
    # kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    # for train_index, val_index in kfold.split(X, numerical_y):
    #     counter = counter + 1
    #     X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    #     y_train, y_val = onehot_y.iloc[train_index], onehot_y.iloc[val_index]
    #
    #     """ ================================================================================================="""
    #     x_train, x_val = X_train.values, X_val.values
    #     X_train, X_val = x_train.reshape((x_train.shape[0], 1, x_train.shape[1])), x_val.reshape((x_val.shape[0], 1,x_val.shape[1]))

        # ERNN.ERNN_direct_training(hidden_units=64, num_class=10, prototypes=64, nu=0.5, is_train=False,
        #                         evi_filepath='pickleJarNB15/KFOLD/ERNN/proto64_dense32_random_KFOLD%d' % counter,
        #                         x_train=X_train, y_train=y_train, x_test=X_val, y_test=y_val,output_confusion_matrix=1)
        # ERNN.RNN(hidden_units=64, num_class=10,model_filepath='pickleJarNB15/KFOLD/RNN/dense32_random_KFOLD%d' % counter,
        #          is_train=True, x_train=X_train, y_train=y_train, x_test=X_val, y_test=y_val, output_confusion_matrix=1)
    #     ELSTM.pLSTM(hidden_units=60, num_class=10, is_train=True,
    #                 model_filepath='pickleJarNB15/KFOLD/LSTM/dense64_random_KFOLD%d' % counter,
    #                 x_train=X_train, y_train=y_train, x_test=X_val, y_test=y_val, output_confusion_matrix=1)
    #     ELSTM.ELSTM_direct_training(hidden_units=60, num_class=10, prototypes=70, nu=0.5, is_train=True,
    #                                 evi_filepath='pickleJarNB15/KFOLD/ELSTM/dense64_noalpha_random_KFOLD%d' % counter,
    #                                 x_train=X_train, y_train=y_train, x_test=X_val, y_test=y_val,
    #                                 output_confusion_matrix=1)
    #     """ ================================================================================================="""
        # ECNN_UNSW_NB15.CNN(data_WIDTH=50, num_class=10,is_train=True,
        #                    model_filepath='pickleJarNB15/KFOLD/CNN/random_KFOLD%d' % counter,
        #                    x_train=X_train, y_train=y_train, x_test=X_val, y_test=y_val,output_confusion_matrix=1)

        # ECNN_UNSW_NB15.ECNN_direct_training(data_WIDTH=50, num_class=10, prototypes=40, nu=0.5, flatten_size=64, is_train=True,
        #                                     evi_filepath='pickleJarNB15/KFOLD/ECNN_direct/noalpha_random_proto40_nu0.5_KFOLD%d' % counter,
        #                                     epochs=90,  x_train=X_train, y_train=y_train, x_test=X_val, y_test=y_val,output_confusion_matrix=1)
    # a2 = r2 = p2 = f2 = 0
    # AU_record = []
    # SVR_record = []
    # for counter in [1, 2, 3, 4]:
    #     X_np, X_test_np = X.values, X_test.values
    #     X_reshape, X_test_reshape = X_np.reshape((X_np.shape[0], 1, X_np.shape[1])), X_test_np.reshape((X_test_np.shape[0], 1, X_test_np.shape[1]))
    #
    #     acc2, recall2, prec2, fs2 = \
    #         ELSTM.pLSTM(hidden_units=60, num_class=10, is_train=False,
    #                     model_filepath='pickleJarNB15/KFOLD/LSTM/dense64_random_KFOLD%d' % counter,
    #                     x_train=X_reshape, y_train=y, x_test=X_reshape, y_test=y, output_confusion_matrix=1)
    #     a2 = a2 + acc2; r2 = r2 + recall2; p2 = p2 + prec2; f2 = f2 + fs2
    # print('Performance on UNSW-NB15')
    # print("accuracy: %f   ; recall: %f   ; precision: %f   ; f1_score: %f" % (a2 / 4, r2 / 4, p2 / 4, f2 / 4))
    #     AU, SVR = check_Model_in_Selective_SV(num_class=10, number_act_set=1023, proto=64,
    #                                 filepath='pickleJarNB15/KFOLD/ERNN/proto64_dense32_random_KFOLD%d' % counter,
    #                                 x_test=X_test_reshape, numerical_y_test=numerical_y_test)
    #     AU_record.append(AU)
    #     SVR_record.append(SVR)
    # print("AU:" + str(AU_record))
    # print("SV Rate:" + str(SVR_record))
    # AU_np = np.array(AU_record)
    # SVR_np = np.array(SVR_record)
    # AU_mean = np.mean(AU_np, axis=0)
    # SVR_np = np.mean(SVR_np, axis=0)
    # print("AU mean:" + str(AU_mean))
    # print("SV Rate mean:" + str(SVR_np))
    #
    # check_Model_in_Selective_SV(num_class=5, number_act_set=31, proto=0,
    #                             filepath='D:', x_test=1, numerical_y_test=1)
