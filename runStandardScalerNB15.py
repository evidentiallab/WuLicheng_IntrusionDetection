import numpy as np
import random
import tensorflow as tf
import pandas as pd
from disusedModel import ONEDCNN_NB15
import model
import model.ECNN_UNSW_NB15 as ECNN
import model.Set_valued as sv
import model.ERNN as ERNN
import model.ELSTM as ELSTM
from scipy.stats import chi2_contingency
import os


def seed_tensorflow(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)


def mcnemar_test(y_true, y_pred1, y_pred2):
    correct1 = np.sum(y_true == y_pred1)
    correct2 = np.sum(y_true == y_pred2)
    wrong1 = len(y_pred1) - correct1
    wrong2 = len(y_pred2) - correct2
    table = np.array([[correct1, wrong1], [correct2, wrong2]])
    chi2, p, dof, expected = chi2_contingency(table)
    print('p-value: ' + str(p))
    return p


def load_dataset(split, is_reshape):
    if split == "official":
        X_train = pd.read_csv('D:/NB15dataset/reselect_X_train0.csv', header=0).values
        X_test = pd.read_csv('D:/NB15dataset/reselect_X_test0.csv', header=0).values

        y = pd.read_csv('D:/NB15dataset/concat_label_onehot.csv', header=0)
        y_train = y.iloc[0:175341].values
        y_test = y.iloc[175341:].values
        numerical_y_test = [np.argmax(i) for i in y_test]
        print("official split")

    elif split == "random":
        X_train = pd.read_csv('D:/NB15dataset/reselect_X_train1.csv', header=0).values
        X_test = pd.read_csv('D:/NB15dataset/reselect_X_test1.csv', header=0).values

        y_train = pd.read_csv('D:/NB15dataset/reselect_y_train1.csv', header=0).values
        y_test = pd.read_csv('D:/NB15dataset/reselect_y_test1.csv', header=0).values
        numerical_y_test = [np.argmax(i) for i in y_test]
        print("random split")

    if is_reshape == True:
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    return X_train, X_test, y_train, y_test, numerical_y_test


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    seed_tensorflow(1)

    """ ------------------------ official split ------------------------ """
    X_train, X_test, y_train, y_test, numerical_y_test = load_dataset("official", is_reshape = False)
    # ECNN.CNN(data_WIDTH=50, num_class=10,
    #          model_filepath='pickleJarNB15/biCNN_SSNB15_official',
    #          is_train=0, is_load=1,
    #          x_train=X_train, y_train=y_train,
    #          x_test=X_test, y_test=y_test,
    #          output_confusion_matrix=1)

    # ECNN.ECNN(data_WIDTH=50, num_class=10, flatten_size=512,
    #           prototypes=50, nu=0.0,
    #           model_filepath='pickleJarNB15/biCNN_SSNB15_official',
    #           mid_filepath='pickleJarNB15/ECNN/mid/SSNB15_official_proto50_nu0.0',
    #           evi_filepath='pickleJarNB15/ECNN/SSNB15_official_proto50_nu0.0',
    #           load_and_train=0, is_load=1,
    #           x_train=X_train, y_train=y_train,
    #           x_test=X_test, y_test=y_test,
    #           output_confusion_matrix=1)
    # num_class = 10
    # class_set = list(range(num_class))
    # act_set = sv.PowerSets(class_set, no_empty=True, is_sorted=True)
    # m = 0
    # for i in [0,1,2,3,4]:
    #     UM = sv.utility_mtx(num_class, act_set=act_set, class_set=class_set, tol_i=i, m=m)
    #     ECNN.CNN_SV(data_WIDTH=50, num_class=10, number_act_set=1023, tol=0.1 * i + 0.5 + m,
    #                 utility_matrix=UM, is_load=1, nu=0.0,
    #                 filepath='pickleJarNB15/biCNN_SSNB15_official',
    #                 act_set=act_set, x_test=X_test, numerical_y_test=numerical_y_test)


    # for hidden_units in [20,40,60,80,100]:
    # y_pred1 = ERNN.bilayer_RNN(hidden_units=80,num_class=10,
    #                     model_filepath=
    #                     'pickleJarRNN/vanillaRNN/reselect_feature_NB15/SSNB15_officialSplit_hiddenunits80',
    #                     is_train=0, is_load=1,
    #                     x_train=X_train0, y_train=y_train0,
    #                     x_test=X_test0, y_test=y_test0,
    #                     pic_filepath_loss='',pic_filepath_acc='',
    #                     output_confusion_matrix=1)
    # for hidden_units in [20, 40, 60, 80, 100]:
    # y_pred1 = ONEDCNN_NB15.bilayer_LSTM_api(hidden_units=60,num_class=10,
    #                                   model_filepath=
    #                                   'pickleJarRNN/LSTM/bilayerLSTM/reselect_feature_NB15SSNB15_officialSplit_hiddenunits60',
    #                                   is_train=0, is_load=1,
    #                                   x_train=X_train0, y_train=y_train0,
    #                                   x_test=X_test0, y_test=y_test0,
    #                                   pic_filepath_loss='', pic_filepath_acc='',
    #                                   output_confusion_matrix=1)
    # for proto0 in [10,20,30,40,50,60,70,80,90,100]:
    #     nu = 0.0
    # for nu in [0.0,0.1,0.2,0.3,0.4,0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    #     proto0 = 100
    # y_pred2 = ERNN.ERNN(hidden_units=80, num_class=10, prototypes=30, nu=0.5,
    #             model_filepath='pickleJarRNN/vanillaRNN/reselect_feature_NB15/SSNB15_officialSplit_hiddenunits80',
    #             mid_filepath='pickleJarNB15/ERNN/mid/SSNB15_officialSplit_proto30_nu0.5',
    #             evi_filepath='pickleJarNB15/ERNN/SSNB15_officialSplit_proto30_nu0.5',
    #             load_and_train=0, is_load=1,
    #             x_train=X_train0, y_train=y_train0,
    #             x_test=X_test0, y_test=y_test0,
    #             output_confusion_matrix=1)

    # y_pred2 = ONEDCNN_NB15.ELSTM1_api(hidden_units=80, num_class=10, prototypes=100,nu=0.0,
    #                         model_filepath='pickleJarRNN/LSTM/bilayerLSTM/reselect_feature_NB15SSNB15_officialSplit_hiddenunits80',
    #                         mid_filepath='pickleJarNB15/ELSTM/mid/SSNB15_officialSplit_proto100_nu0.0',
    #                         evi_filepath='pickleJarNB15/ELSTM/SSNB15_officialSplit_proto100_nu0.0' ,
    #                         load_and_train=0, is_load=1,
    #                         x_train=X_train0, y_train=y_train0,
    #                         x_test=X_test0, y_test=y_test0,
    #                         output_confusion_matrix=1)
    # num_class = 10
    # class_set = list(range(num_class))
    # act_set = sv.PowerSets(class_set, no_empty=True, is_sorted=True)
    # m = 0
    # for i in [0,1,2,3,4]:
    # # for i in [4]:
    #     m = 0.05
    #     UM = sv.utility_mtx(num_class, act_set=act_set, class_set=class_set, tol_i=i, m=m)
    #     ERNN.ERNN_set_value(hidden_units=80, num_class=10, number_act_set=1023, act_set=act_set, tol=0.5+0.1*i+m,
    #                         nu=0.5,prototypes=30, utility_matrix=UM,
    #                         is_load=1, filepath='pickleJarNB15/ERNN/SSNB15_officialSplit_proto30_nu0.5',
    #                         x_test=X_test0, numerical_y_test=numerical_y_test0)
    # ONEDCNN_NB15.ELSTM_set_value_api(hidden_units=80, num_class=10, number_act_set=1023, act_set=act_set,tol=0.5+0.1*i+m,
    #                         nu=0.0,prototypes=100,
    #                         utility_matrix=UM, is_load=1,
    #                         filepath='pickleJarNB15/ELSTM/SSNB15_officialSplit_proto100_nu0.0',
    #                         x_test=X_test0, numerical_y_test=numerical_y_test0)
    #     ONEDCNN_NB15.set_value_api(data_WIDTH=50, num_class=10, number_act_set=1023, tol=0.1 * i + 0.5 + m,
    #                            utility_matrix=UM, is_load=1,
    #                            filepath='pickleJarNB15/ECNN/SSNB15_official_proto50_nu0.0',
    #                            act_set=act_set, nu=0.0, prototypes=50,
    #                            x_test=X_test, numerical_y_test=numerical_y_test)

    # tb = mcnemar_table(y_target=y_test0.argmax(axis=1), y_model1=y_pred1, y_model2=y_pred2)
    # mcnemar_test(y_true=y_test0.argmax(axis=1), y_pred1=y_pred1, y_pred2=y_pred2)
    # chi2, p = mcnemar(ary=tb, corrected=True)
    # print("p-value: " + str(p))

    """"" ================================== random split ==================================  """
    X_train, X_test, y_train, y_test, numerical_y_test = load_dataset("random", is_reshape=True)

    # ECNN.Test_CNN_SV(data_WIDTH=50,num_class=10, number_act_set=1023, nu=0.5,
    #                 filepath='pickleJarNB15/biCNN_SSNB15_random', x_test=X_test, numerical_y_test=numerical_y_test)
    # ERNN.Test_RNN_SV(hidden_units=100, data_WIDTH=50, num_class=10, number_act_set=1023, nu=0.5,
    #                  filepath='pickleJarNB15/RNN_SSNB15_random_hiddenunits100', x_test=X_test, numerical_y_test=numerical_y_test)
    ELSTM.Test_LSTM_SV(hidden_units=100, data_WIDTH=50, num_class=10, number_act_set=1023, nu=0.5,
                     filepath='pickleJarNB15/LSTM_SSNB15_random_hiddenunits100', x_test=X_test, numerical_y_test=numerical_y_test)