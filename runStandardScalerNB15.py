import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import ECNN_NSL
from model import ECNN_Conv2maxpool2
from model import ONEDCNN_NB15
import model.Set_valued as sv
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold,StratifiedKFold
import os


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    """no split without ROS """
    concat_df = pd.read_csv('D:/NB15dataset/standardscale_concat.csv',header=0)
    train_no_split = concat_df.iloc[:175341, 0:43].values
    test_no_split = concat_df.iloc[175341:, 0:43].values
    # print(train_no_split)
    # print(test_no_split.shape)
    concat_data = concat_df.values

    concat_label = pd.read_csv('D:/NB15dataset/concat_label_onehot.csv',header=0)
    train_label_no_split = concat_label.iloc[:175341].values
    test_label_no_split = concat_label.iloc[175341:].values
    numerical_test_label_no_split = [np.argmax(i) for i in test_label_no_split]
    # print(train_label_no_split.shape)
    # print(test_label_no_split)
    # print(numerical_test_label_no_split)

    ONEDCNN_NB15.probabilistic_CNN_api(data_WIDTH=43,num_class=10,is_train=0,is_load=1,
                                       model_filepath='pickleJar/Conv1D/SSNB15_43features_no_split_no_ROS_3layers',
                                       x_train=train_no_split,y_train=train_label_no_split,
                                       x_test=test_no_split, y_test=test_label_no_split,
                                       pic_fp0='',pic_fp1='',
                                       output_confusion_matrix=1)

    """split without ROS """
    train,test,train_l,test_l = train_test_split(concat_df.iloc[:,0:43],concat_label,test_size=0.3,random_state=1,shuffle=True)
    train,test,train_l,test_l = train.values,test.values,train_l.values,test_l.values
    numerical_test_l = [np.argmax(i) for i in test_l]
    # print(train.shape)
    # print(test.shape)
    # print(train_l.shape)
    # print(test_l.shape)
    ONEDCNN_NB15.probabilistic_CNN_api(data_WIDTH=43, num_class=10, is_train=0, is_load=1,
                                       model_filepath='pickleJar/Conv1D/SSNB15_43features_no_ROS_3layers',
                                       x_train=train, y_train=train_l,
                                       x_test=test, y_test=test_l,
                                       pic_fp0='', pic_fp1='',
                                       output_confusion_matrix=1)

    """ with ROS """
    # print(Counter(concat_df['attack_cat']))
    ros = RandomOverSampler(sampling_strategy={'Normal': 93000, 'Generic': 58871, 'Exploits': 44525,
                                               'Fuzzers': 24246, 'DoS': 16353, 'Reconnaissance': 13987,
                                               'Analysis': 8000, 'Backdoor': 8000, 'Shellcode': 5000, 'Worms': 3000},random_state=1)
    x_resampled,y_resampled = ros.fit_resample(concat_df.iloc[:,0:43],concat_df['attack_cat'])
    # y_resampled = y_resampled.reshape(1,-1)
    # print(Counter(y_resampled))

    one_hot = OneHotEncoder()
    y_resampled_df = pd.DataFrame(y_resampled)
    y_resampled_onehot = pd.get_dummies(y_resampled_df,prefix="label")
    # print(y_resampled_onehot)
    train_ROS, test_ROS, train_ROS_l, test_ROS_l = train_test_split(x_resampled, y_resampled_onehot, test_size=0.3, random_state=1)
    train_ROS, test_ROS, train_ROS_l, test_ROS_l = train_ROS.values, test_ROS.values, train_ROS_l.values, test_ROS_l.values
    numerical_test_ROS_l = [np.argmax(i) for i in test_ROS_l]
    # print(train_ROS.shape)
    # print(test_ROS.shape)
    # print(train_ROS_l.shape)
    # print(test_ROS_l)
    # print(numerical_test_ROS_l)
    ONEDCNN_NB15.probabilistic_CNN_api(data_WIDTH=43, num_class=10, is_train=0, is_load=1,
                                       model_filepath='pickleJar/Conv1D/SSNB15_43features_ROS_3layers',
                                       x_train=train_ROS, y_train=train_ROS_l,
                                       x_test=test_ROS, y_test=test_ROS_l,
                                       pic_fp0='', pic_fp1='',
                                       output_confusion_matrix=1)


    train1_lstm = train_no_split.reshape(train_no_split.shape[0],1,train_no_split.shape[1])
    test1_lstm = test_no_split.reshape(test_no_split.shape[0],1,test_no_split.shape[1])

    train2_lstm = train.reshape(train.shape[0], 1, train.shape[1])
    test2_lstm = test.reshape(test.shape[0], 1, test.shape[1])

    train3_lstm = train_ROS.reshape(train_ROS.shape[0], 1, train_ROS.shape[1])
    test3_lstm = test_ROS.reshape(test_ROS.shape[0], 1, test_ROS.shape[1])

    # for hidden_u in [20,40,50,60,80]:
    # hidden_u1 = 40
    # ONEDCNN_NB15.bilayer_LSTM_api(hidden_units=hidden_u1, num_class=10,
    #                               model_filepath='pickleJarRNN/LSTM/bilayerLSTM/SSNB15_nosplit_hiddenunits%d'%hidden_u,
    #                               is_train=0, is_load=0,
    #                               x_train=train1_lstm, y_train=train_label_no_split,
    #                               x_test=test1_lstm, y_test=test_label_no_split,
    #                               pic_filepath_loss='', pic_filepath_acc='',
    #                               output_confusion_matrix=1)
    # ONEDCNN_NB15.bilayer_LSTM_api(hidden_units=hidden_u, num_class=10,
    #                                   model_filepath='pickleJarRNN/LSTM/bilayerLSTM/SSNB15_split_hiddenunits%d' % hidden_u,
    #                                   is_train=0, is_load=0,
    #                                   x_train=train2_lstm, y_train=train_l,
    #                                   x_test=test2_lstm, y_test=test_l,
    #                                   pic_filepath_loss='', pic_filepath_acc='',
    #                                   output_confusion_matrix=1)
    # ONEDCNN_NB15.bilayer_LSTM_api(hidden_units=hidden_u, num_class=10,
    #                                   model_filepath='pickleJarRNN/LSTM/bilayerLSTM/SSNB15_ROS_hiddenunits%d' % hidden_u,
    #                                   is_train=0, is_load=0,
    #                                   x_train=train3_lstm, y_train=train_ROS_l,
    #                                   x_test=test3_lstm, y_test=test_ROS_l,
    #                                   pic_filepath_loss='', pic_filepath_acc='',
    #                                   output_confusion_matrix=1)
    # for proto1 in [10,20,30,40,50,60,70,80,90,100]:
    # for nu in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    #     proto1 = 70
    #     ONEDCNN_NB15.ELSTM1_api(hidden_units=40, num_class=10, prototypes=proto1,nu=nu,
    #                         model_filepath='pickleJarRNN/LSTM/bilayerLSTM/SSNB15_nosplit_hiddenunits40',
    #                         mid_filepath='pickleJarRNN/LSTM/ELSTM_NB15/mid/SSNB15_nosplit_proto%d_nu%s'%(proto1,str(nu)),
    #                         evi_filepath='pickleJarRNN/LSTM/ELSTM_NB15/SSNB15_nosplit_proto%d_nu%s'%(proto1,str(nu)),
    #                         load_and_train=0, is_load=1,
    #                         x_train=train1_lstm, y_train=train_label_no_split,
    #                         x_test=test1_lstm, y_test=test_label_no_split,
    #                         output_confusion_matrix=1)

    # for proto2 in [10,20,30,40,50,60,70,80,90,100]:
    # for nu in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    #     proto2 = 80
    #     ONEDCNN_NB15.ELSTM1_api(hidden_units=60, num_class=10, prototypes=proto2,nu=nu,
    #                         model_filepath='pickleJarRNN/LSTM/bilayerLSTM/SSNB15_split_hiddenunits60',
    #                         mid_filepath='pickleJarRNN/LSTM/ELSTM_NB15/mid/SSNB15_split_proto%d_nu%s'%(proto2,str(nu)),
    #                         evi_filepath='pickleJarRNN/LSTM/ELSTM_NB15/SSNB15_split_proto%d_nu%s'%(proto2,str(nu)),
    #                         load_and_train=0, is_load=1,
    #                         x_train=train2_lstm, y_train=train_l,
    #                         x_test=test2_lstm, y_test=test_l,
    #                         output_confusion_matrix=1)

    # for proto3 in [10,20,30,40,50,60,70,80,90,100]:
    # for nu in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    #     proto3 = 50
    #     ONEDCNN_NB15.ELSTM1_api(hidden_units=80, num_class=10, prototypes=proto3,nu=nu,
    #                         model_filepath='pickleJarRNN/LSTM/bilayerLSTM/SSNB15_ROS_hiddenunits80',
    #                         mid_filepath='pickleJarRNN/LSTM/ELSTM_NB15/mid/SSNB15_ROS_proto%d_nu%s'%(proto3,str(nu)),
    #                         evi_filepath='pickleJarRNN/LSTM/ELSTM_NB15/SSNB15_ROS_proto%d_nu%s'%(proto3,str(nu)),
    #                         load_and_train=0, is_load=1,
    #                         x_train=train3_lstm, y_train=train_ROS_l,
    #                         x_test=test3_lstm, y_test=test_ROS_l,
    #                         output_confusion_matrix=1)

    # num_class = 10
    # class_set = list(range(num_class))
    # act_set = sv.PowerSets(class_set, no_empty=True, is_sorted=True)
    # for i in [4]:
    #     # i = 4
    #     UM = sv.utility_mtx(num_class, act_set=act_set, class_set=class_set, tol_i=i, m=0.05)
        # ONEDCNN_NB15.ELSTM_set_value_api(hidden_units=40,num_class=10, number_act_set=1023, act_set=act_set,utility_matrix=UM,
        #                                  nu=0.0, tol=0.1*i+0.5+0.05,prototypes=70,is_load=1,
        #                                  filepath='pickleJarRNN/LSTM/ELSTM_NB15/SSNB15_nosplit_proto70_nu0',
        #                                  x_test=test1_lstm, numerical_y_test=numerical_test_label_no_split)

        # ONEDCNN_NB15.ELSTM_set_value_api(hidden_units=60, num_class=10, number_act_set=1023, act_set=act_set,
        #                                  utility_matrix=UM,
        #                                  nu=0.4, tol=0.1 * i+0.5+0.05, prototypes=80, is_load=1,
        #                                  filepath='pickleJarRNN/LSTM/ELSTM_NB15/SSNB15_split_proto80_nu0.4',
        #                                  x_test=test2_lstm, numerical_y_test=numerical_test_l)
        #
        # ONEDCNN_NB15.ELSTM_set_value_api(hidden_units=80, num_class=10, number_act_set=1023, act_set=act_set,
        #                                  utility_matrix=UM,
        #                                  nu=0.2, tol=0.1 * i+0.5+0.05, prototypes=50, is_load=1,
        #                                  filepath='pickleJarRNN/LSTM/ELSTM_NB15/SSNB15_ROS_proto50_nu0.2',
        #                                  x_test=test2_lstm, numerical_y_test=numerical_test_l)

