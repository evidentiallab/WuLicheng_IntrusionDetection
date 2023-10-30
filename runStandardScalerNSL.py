import numpy as np
import random
import pandas as pd
import tensorflow as tf
from disusedModel import ONEDCNN_NB15
import model.ECNN_NSL_KDD as ECNN
import model.ERNN as ERNN
import model.ELSTM as ELSTM

import os


def seed_tensorflow(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)


def load_dataset(split, is_reshape):
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
        X_train = pd.read_csv('D:/IDdataset/X_train1.csv', header=0).values
        X_test = pd.read_csv('D:/IDdataset/X_test1.csv', header=0).values

        y_train = pd.read_csv('D:/IDdataset/y_train1.csv', header=0).values
        y_test = pd.read_csv('D:/IDdataset/y_test1.csv', header=0).values
        numerical_y_test = [np.argmax(i) for i in y_test]
        print("random split")

    if is_reshape == True:
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    return X_train, X_test, y_train, y_test, numerical_y_test


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    seed_tensorflow(1)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)

    """ Feature selection and visualization """
    # onehot_concat_df = pd.read_csv('D:/IDdataset/processedNSL/concat_onehot_ordered.csv', header=0)
    # onehot_concat_df = onehot_concat_df.drop("difficulty_level",axis=1)
    #
    # X = onehot_concat_df.iloc[:,0:129]
    # y = onehot_concat_df.iloc[:,129:]
    # skb = SelectKBest(score_func=chi2, k=40)
    # # print(new_X.shape)
    # newX = skb.fit_transform(X,y)
    # # new_concat_df.to_csv('D:/IDdataset/processedNSL/new_X_concat.csv',index=False,header=False)
    #
    # df_scores = pd.DataFrame(skb.scores_)
    # df_columns = pd.DataFrame(X.columns)
    # df_feature_scores = pd.concat([df_columns, df_scores], axis=1)
    # df_feature_scores.columns = ['Feature', 'Score']
    #
    # sss = df_feature_scores.sort_values(by='Score', ascending=False)
    # print(sss)

    # newX = pd.read_csv('D:/IDdataset/processedNSL/new_X_concat.csv', header=0)

    """ official split """
    # y = pd.read_csv('D:/IDdataset/reprocess-524/concat_onehot_ordered.csv', header=0)
    # y = y.iloc[:,129:134]
    # y_train0 = y.iloc[0:125973].values
    # y_test0 = y.iloc[125973:].values
    # numerical_y_test0 = [np.argmax(i) for i in y_test0]


    # print(X_train)
    # print(X_test)
    # print(y_train)
    # print(y_test)
    # sscaler = StandardScaler()
    # X_train0 = sscaler.fit_transform(X_train)
    # X_test0 = sscaler.transform(X_test)
    # np.savetxt('D:/IDdataset/X_train0.csv',X_train0,delimiter=',',fmt='%.5f')
    # np.savetxt('D:/IDdataset/X_test0.csv', X_test0, delimiter=',', fmt='%.5f')

    # X_train0 = pd.read_csv('D:/IDdataset/X_train0.csv',header=0).values
    # X_test0 = pd.read_csv('D:/IDdataset/X_test0.csv', header=0).values

    # X_train0 = X_train0.reshape(X_train0.shape[0], 1, X_train0.shape[1])
    # X_test0 = X_test0.reshape(X_test0.shape[0], 1, X_test0.shape[1])

    # SNN.probabilistic_biSNN(data_WIDTH=40, num_class=5,
    #                                 model_filepath='pickleJarNSL/biSNN_SSNSL_official',
    #                                 is_train=0, is_load=1,
    #                                 x_train=X_train0, y_train=y_train0,
    #                                 x_test=X_test0, y_test=y_test0,
    #                                 output_confusion_matrix=1)


    # for nu in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:

    #     proto0 = 40
    #     SNN.evidential_SNN(data_WIDTH=40, num_class=5, prototypes=proto0,nu=nu,flatten_size=64,
    #                        model_filepath='pickleJarNSL/biSNN_SSNSL_official',
    #                        mid_filepath='pickleJarNSL/ESNN/mid/biSNN_SSNSL_official_proto%d_nu%s'%(proto0, str(nu)),
    #                        evi_filepath='pickleJarNSL/ESNN/biSNN_SSNSL_official_proto%d_nu%s'%(proto0, str(nu)),
    #                        load_and_train=0, is_load=1,
    #                        x_train=X_train0, y_train=y_train0,
    #                        x_test=X_test0, y_test=y_test0,
    #                        output_confusion_matrix=1)
    #     ONEDCNN_NB15.evidential_CNN_api(data_WIDTH=40, num_class=5, prototypes=proto0,nu=nu,flatten_size=64,
    #                        model_filepath='pickleJarNSL/biCNN(2FC)_SSNSL_official',
    #                        mid_filepath='pickleJarNSL/ECNN/mid/SSNSL(2FC)_official_proto%d_nu%s'%(proto0, str(nu)),
    #                        evi_filepath='pickleJarNSL/ECNN/SSNSL(2FC)_official_proto%d_nu%s'%(proto0, str(nu)),
    #                        load_and_train=0, is_load=1,
    #                        x_train=X_train0, y_train=y_train0,
    #                        x_test=X_test0, y_test=y_test0,
    #                        output_confusion_matrix=1)

    # num_class = 5
    # class_set = list(range(num_class))
    # act_set = sv.PowerSets(class_set, no_empty=True, is_sorted=True)
    # m = 0.0
    # for i in [0,1,2,3,4]:
    # m = 0.05
    # for i in [4]:
    #     UM = sv.utility_mtx(num_class, act_set=act_set, class_set=class_set, tol_i=i, m=m)
        # ERNN.ERNN_set_value(hidden_units=40, num_class=5, number_act_set=31, act_set=act_set, tol=0.5+0.1*i+m,
        #                     nu=0, prototypes=80, utility_matrix=UM,
        #                     is_load=1, filepath='pickleJarRNN/vanillaRNN/ERNN/SSNSL_officialSplit_proto80_nu0',
        #                     x_test=X_test0, numerical_y_test=numerical_y_test0)
        # ONEDCNN_NB15.ELSTM_set_value_api(hidden_units=60, num_class=5, number_act_set=31, act_set=act_set, utility_matrix=UM,
        #                     nu=0.0, tol=0.1 * i + 0.5 + m, prototypes=20,
        #                     is_load=1, filepath='pickleJarRNN/LSTM/ELSTM/SSNSL_officialSplit_proto20_nu0.0',
        #                     x_test=X_test0, numerical_y_test=numerical_y_test0)

        # ONEDCNN_NB15.set_value_api(data_WIDTH=40, num_class=5, number_act_set=31, tol=0.1 * i + 0.5 + m,
        #                            utility_matrix=UM, is_load=1, filepath='pickleJarNSL/ECNN/SSNSL(2FC)_official_proto20_nu0.0',
        #                            act_set=act_set, nu=0.0,  prototypes=20,
        #                            x_test=X_test0, numerical_y_test=numerical_y_test0)


    """ random split """
    X_train, X_test, y_train, y_test, numerical_y_test = load_dataset(split='random', is_reshape=False)
    # ECNN.CNN(data_WIDTH=40, num_class=5, is_train=0, is_load=1,
    #          model_filepath='pickleJarNSL/biCNN(2FC)_SSNSL_random',
    #          x_train=X_train, y_train=y_train,
    #          x_test=X_test, y_test=y_test, output_confusion_matrix=1)
    #
    # ECNN.Test_CNN_SV(data_WIDTH=40,num_class=5, number_act_set=31, nu=0.5,
    #                 filepath='pickleJarNSL/biCNN(2FC)_SSNSL_random', x_test=X_test, numerical_y_test=numerical_y_test)

    # ERNN.RNN(hidden_units=40, num_class=5, model_filepath='pickleJarRNN/vanillaRNN/SSNSL_randomSplit_hiddenunits40',
    #          is_train=0, is_load=1, output_confusion_matrix=1,
    #          x_train=X_train, y_train=y_train,
    #          x_test=X_test, y_test=y_test)
    # ERNN.Test_RNN_SV(hidden_units=40, data_WIDTH=40, num_class=5, number_act_set=31, nu=0.5,
    #                  filepath='pickleJarRNN/vanillaRNN/SSNSL_randomSplit_hiddenunits40', x_test=X_test, numerical_y_test=numerical_y_test)
    # ELSTM.Test_LSTM_SV(hidden_units=60, data_WIDTH=40, num_class=5, number_act_set=31, nu=0.5,
    #                  filepath='pickleJarRNN/LSTM/bilayerLSTM/SSNSL_randomSplit_hiddenunits60', x_test=X_test, numerical_y_test=numerical_y_test)
    ECNN.ECNN(data_WIDTH=40, num_class=5, prototypes=40, nu=0.5,
            model_filepath='pickleJarNSL/biCNN(2FC)_SSNSL_random',
            evi_filepath='pickleJarNSL/ECNN/mid/SSNSL(2FC)_random_proto40_nu0.5',
            mid_filepath='pickleJarNSL/ECNN/SSNSL(2FC)_random_proto40_nu0.5',
            flatten_size=64, load_and_train=1, is_load=0,
            x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test, output_confusion_matrix=1)











