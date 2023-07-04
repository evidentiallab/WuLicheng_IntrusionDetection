import numpy as np
import pandas as pd
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from model import ECNN_NSL
from sklearn import preprocessing
from model import ECNN_Conv2maxpool2
from model import ERNN
from model import LSTM
from model import Set_valued
import model.Set_valued as sv
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import OneHotEncoder
import os


def seed_tensorflow(seed=42):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # seed_tensorflow(seed=42)
    concat_df = pd.read_csv('D:/IDdataset/feat122/concat.csv',header=0)
    train = concat_df.iloc[:125973,0:122].values
    train_label = concat_df.iloc[:125973,122:].values
    test = concat_df.iloc[125973:,0:122].values
    test_label = concat_df.iloc[125973:,122:].values
    # print(train.shape)
    # print(train_label.shape)
    # print(test.shape)
    # print(test_label.shape)
    # reshape input to [samples, time steps, features]
    x_train = np.reshape(train,(train.shape[0],1,train.shape[1]))
    x_test = np.reshape(test,(test.shape[0],1,test.shape[1]))

    x_train_122 = np.reshape(train, (train.shape[0], train.shape[1],1))
    x_test_122 = np.reshape(test, (test.shape[0], test.shape[1],1))


    """ check learning rate """
    # for num_units in [40,64,80,100,120]:
    #     LSTM.probabilistic_LSTM(hidden_units=num_units, num_class=5, lr=0.1,
    #                             model_filepath='pickleJarRNN/LSTM/NSL_hiddenunits%d_timesteps122_lr0.01' % num_units,
    #                             is_train=0, is_load=1,
    #                             x_train=x_train_122, y_train=train_label,
    #                             x_test=x_test_122, y_test=test_label,
    #                             pic_filepath_loss='', pic_filepath_acc='',
    #                             output_confusion_matrix=1)
    #     LSTM.probabilistic_LSTM(hidden_units=num_units, num_class=5, lr=0.1,
    #                             model_filepath='pickleJarRNN/LSTM/NSL_hiddenunits%d_timesteps1_lr0.01' % num_units,
    #                             is_train=0, is_load=1,
    #                             x_train=x_train, y_train=train_label,
    #                             x_test=x_test, y_test=test_label,
    #                             pic_filepath_loss='', pic_filepath_acc='',
    #                             output_confusion_matrix=1)
    #     print('************hidden units: ' + str(num_units) + '************')

    # for num_units in [40,64,80,100,120]:
    #
    #     LSTM.probabilistic_LSTM(hidden_units=num_units, num_class=5, lr=0.05,
    #                             model_filepath='pickleJarRNN/LSTM/NSL_hiddenunits%d_timesteps122_SGDlr0.1' % num_units,
    #                             is_train=0, is_load=1,
    #                             x_train=x_train_122, y_train=train_label,
    #                             x_test=x_test_122, y_test=test_label,
    #                             pic_filepath_loss='', pic_filepath_acc='',
    #                             output_confusion_matrix=1)
    #     LSTM.probabilistic_LSTM(hidden_units=num_units, num_class=5, lr=0.1,
    #                             model_filepath='pickleJarRNN/LSTM/NSL_hiddenunits%d_timesteps1_SGDlr0.1' % num_units,
    #                             is_train=0, is_load=1,
    #                             x_train=x_train, y_train=train_label,
    #                             x_test=x_test, y_test=test_label,
    #                             pic_filepath_loss='', pic_filepath_acc='',
    #                             output_confusion_matrix=1)
    #     print('************hidden units: ' + str(num_units) + '************')
    """ check learning rate,bilayer LSTM with Adam """
    # for num_units in [16,32,64,128]:
    #     LSTM.bilayer_LSTM(hidden_units=num_units, num_class=5, lr=0.05,
    #                       model_filepath='pickleJarRNN/LSTM/bilayerLSTM/NSL_hiddenunits%d_timesteps122_adam' % num_units,
    #                       is_train=0, is_load=1,
    #                       x_train=x_train_122, y_train=train_label,
    #                       x_test=x_test_122, y_test=test_label,
    #                       pic_filepath_loss='', pic_filepath_acc='',
    #                       output_confusion_matrix=1)
    #     LSTM.bilayer_LSTM(hidden_units=num_units, num_class=5, lr=0.1,
    #                       model_filepath='pickleJarRNN/LSTM/bilayerLSTM/NSL_hiddenunits%d_timesteps1_adam' % num_units,
    #                       is_train=0, is_load=1,
    #                       x_train=x_train, y_train=train_label,
    #                       x_test=x_test, y_test=test_label,
    #                       pic_filepath_loss='', pic_filepath_acc='',
    #                       output_confusion_matrix=1)
    #     print('************hidden units: ' + str(num_units) + '************')

    # LSTM.singlelayer_LSTM_sequential(hidden_units=62, num_class=5, lr=0.05,
    #                       model_filepath='pickleJarRNN/LSTM/singlelayerLSTM(seq)/NSL_hiddenunits62_timesteps122_adam',
    #                       is_train=1, is_load=0,
    #                       x_train=x_train_122, y_train=train_label,
    #                       x_test=x_test_122, y_test=test_label,
    #                       pic_filepath_loss='', pic_filepath_acc='',
    #                       output_confusion_matrix=1)

    """ Direct training EVI_LSTM """
    # LSTM.singlelayer_LSTM_sequential(hidden_units=80, num_class=5, lr=0.05,
    #                                  # model_filepath='E:/Obj/pickleJarRNN/LSTM/singlelayerLSTM/NSL_hiddenunits80_timesteps122_adam',
    #                                  model_filepath='pickleJarRNN/LSTM/singlelayerLSTM(seq)/NSL_hiddenunits80_timesteps122_adam',
    #                                  is_train=0, is_load=0,
    #                                  x_train=x_train_122, y_train=train_label,
    #                                  x_test=x_test_122, y_test=test_label,
    #                                  pic_filepath_loss='', pic_filepath_acc='',
    #                                  output_confusion_matrix=1)

    # for prototype_num in [30,40,50,60,70,80,90,100,120]:
    #     nu = 0.1
    # for nu in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
    #     prototype_num = 80
    #     # LSTM.evidential_LSTM_sequential(hidden_units=80, num_class=5, nu=nu, prototypes=prototype_num,
    #     #                             is_train=4, is_load=1,
    #     #                             evi_filepath='pickleJarRNN/LSTM/EVI_LSTM(seq)/timestep122/NSL_proto%d_nu%s' % (prototype_num, str(nu)),
    #     #                             x_train=x_train_122, y_train=train_label,
    #     #                             x_test=x_test_122, y_test=test_label,
    #     #                             output_confusion_matrix=1)
    #     LSTM.evidential_LSTM_sequential(hidden_units=62, num_class=5, nu=nu, prototypes=prototype_num,
    #                                     is_train=0,is_load=1,
    #                                     evi_filepath='pickleJarRNN/LSTM/EVI_LSTM(seq)/timestep1/NSL_hidden62_proto%d_nu%s' % (prototype_num, str(nu)),
    #                                     x_train=x_train, y_train=train_label,
    #                                     x_test=x_test, y_test=test_label,
    #                                     output_confusion_matrix=1)
    """ set-valued mission """
    test_label_numerical = pd.read_csv('D:/IDdataset/label/test_label_numerical.csv', header=0)['label'].values
    num_class = 5
    class_set = list(range(num_class))
    act_set = sv.PowerSets(class_set, no_empty=True, is_sorted=True)
    UM = sv.utility_mtx(num_class, act_set=act_set, class_set=class_set, tol_i=4, m=0.0999)
    Set_valued.set_valued_evidential_LSTM(num_class=5, number_act_set=31,hidden_units=62, act_set=act_set,
                                          nu=0.1, tol=0.9,prototypes=40,
                                          utility_matrix=UM, is_load=1,
                                          filepath='pickleJarRNN/LSTM/EVI_LSTM(seq)/timestep1/NSL_hidden62_proto40_nu0.1',
                                          x_train=x_train,
                                          x_test=x_test, numerical_y_test=test_label_numerical)



