import pandas as pd
import matplotlib.pyplot as plt
from model import ECNN_NSL
from model import ECNN_Conv2maxpool2
import model.Set_valued as sv
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import OneHotEncoder

if __name__ == '__main__':

    """ 121features """
    # old_train = pd.read_csv('D:/IDdataset/reprocess-524/train(MinMax).csv',header=0)
    # old_train = old_train.drop(["num_outbound_cmds","land_0","land_1",
    #                             "urgent","num_failed_logins","num_compromised","num_root","num_shells"],axis=1).values
    # old_train_2D = old_train.reshape(old_train.shape[0],11,11)
    #
    # old_test = pd.read_csv('D:/IDdataset/reprocess-524/test(MinMax).csv', header=0)
    # old_test = old_test.drop(["num_outbound_cmds","land_0","land_1",
    #                           "urgent","num_failed_logins","num_compromised","num_root","num_shells"],axis=1).values
    # old_test_2D = old_test.reshape(old_test.shape[0], 11, 11)
    #
    # old_train_label = pd.read_csv('D:/IDdataset/label/train_label.csv',header=0).values
    # old_train_label_numerical = pd.read_csv('D:/IDdataset/label/train_label_numerical.csv',header=0)['label'].values
    #
    # old_test_label = pd.read_csv('D:/IDdataset/label/test_label.csv',header=0).values
    old_test_label_numerical = pd.read_csv('D:/IDdataset/label/test_label_numerical.csv', header=0)['label'].values
    """ 100features """
    train = pd.read_csv('D:/IDdataset/processedNSL/test(129-100).csv',header=None).values
    train_2D = train.reshape(train.shape[0],10,10)
    test = pd.read_csv('D:/IDdataset/processedNSL/test(129-100).csv',header=None).values
    test_2D = test.reshape(test.shape[0],10,10)

    train_label = pd.read_csv('D:/IDdataset/label/train_label.csv',header=0).values
    test_label = pd.read_csv('D:/IDdataset/label/test_label.csv',header=0).values
    # old_test_label_numerical = pd.read_csv('D:/IDdataset/label/no_novelty_test_label_numerical.csv',header=0).values
    # old_train_sub_label = pd.read_csv('D:/IDdataset/label/train_sub_label_onehot.csv', header=0).values
    # old_test_sub_label = pd.read_csv('D:/IDdataset/label/no_novelty_test_sub_label_onehot.csv', header=0).values
    #
    # old_test_sub_label_numerical = pd.read_csv('D:/IDdataset/label/no_novelty_test_sublabel_numerical.csv', header=0)
    # old_test_sub_label_numerical = old_test_sub_label_numerical['numerical'].values
    # print(old_test_sub_label_numerical)

    # proto_acc_df = pd.read_csv('pickleJar/PrototypeGradient_23cls/proto-acc gradient.csv',header=0)
    # for num_proto in [100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180]:
    # for num_proto in [185, 190, 195, 200, 205, 210,215,220,225,230,235,240,245,250]:
    # for num_proto in [121,122,123,124,126,127,128,129]:
    # for nu in [0.0,0.1,0.2,0.3,0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    #     num_proto = 124
    #     acc = ECNN_NSL.evidential_FitNet4(data_WIDTH=10, data_HEIGHT=10, num_class=23, prototypes=num_proto, nu=nu,
    #                             model_filepath='pickleJar/subclass_classification/PR_FN4_NSL', plot_DS_layer_loss=0,
    #                             load_weights=0, train_DS_Layers=0,
    #                             evi_filepath='pickleJar/PrototypeGradient_23cls/EVI_FN4_NSL_proto%d_nu%s' % (num_proto,str(nu)),
    #                             mid_filepath='pickleJar/PrototypeGradient_23cls/mid/FN4_NSL_proto%d_nu%s' % (num_proto,str(nu)),
    #                             x_train=old_train_2D, y_train=old_train_sub_label,
    #                             x_test=old_test_2D, y_test=old_test_sub_label,
    #                             is_load=1, output_confusion_matrix=1)
    #     proto_acc_df.loc[len(proto_acc_df.index)] = [num_proto,acc]
    # proto_acc_df.to_csv('pickleJar/PrototypeGradient_23cls/proto-acc gradient.csv', index=False)
    """ sampling """
    # print(Counter(old_train_label_numerical))
    # ros = RandomOverSampler(sampling_strategy={0: 67343, 1: 45927, 2: 11656, 3: 2000, 4: 6000},random_state=0)
    # x_resampled,y_resampled = ros.fit_resample(old_train,old_train_label_numerical)
    # # y_resampled = y_resampled.reshape(1,-1)
    # # print(Counter(y_resampled))
    # x_resampled_2D = x_resampled.reshape(x_resampled.shape[0], 11, 11)
    # one_hot = OneHotEncoder()
    # y_resampled_df = pd.DataFrame(y_resampled,columns=['label'])
    # y_resampled_onehot = pd.get_dummies(y_resampled_df,columns=['label']).values
    # print(y_resampled_onehot)

    """ train or load probabilistic model """
    # ECNN_NSL.probabilistic_FitNet4(data_WIDTH=11, data_HEIGHT=11, num_class=5,
    #                                is_train=0, is_load=0,
    #                                model_filepath='pickleJar/withoutAE-524/PR_FN4_NSL(ROS2000_6000)',
    #                                pic_filepath_loss='', pic_filepath_acc='',
    #                                x_train=train_2D, y_train=train_label,
    #                                x_test=train_2D, y_test=test_label,
    #                                output_confusion_matrix=1)
    """ train or load evidential model """
    ECNN_NSL.evidential_FitNet4(data_WIDTH=10, data_HEIGHT=10, num_class=5,
                                prototypes=100, nu=0.1,
                                model_filepath='pickleJar/NoNovelty/PR_FN4_NSL',
                                evi_filepath='E:/Obj/pickleJar/NoNovelty/EVI_FN4_NSL_proto100_nu0.1',
                                mid_filepath='E:/Obj/pickleJar/mid/EVI_FN4_NSL_proto100_nu0.1',
                                load_weights=0, train_DS_Layers=0, plot_DS_layer_loss=0,
                                x_train=train_2D, y_train=train_label,
                                x_test=test_2D, y_test=test_label,
                                is_load=1,output_confusion_matrix=1)
    # for num_proto in [20,30,40,50,60,70,80,90,100,110,120,200,400]:
    #     nu = 0.1
    #     ECNN_NSL.evidential_FitNet4(data_WIDTH=11, data_HEIGHT=11, num_class=5,
    #                                 prototypes=num_proto, nu=nu,
    #                                 model_filepath='pickleJar/withoutAE-524/PR_FN4_NSL(ROS2000/6000)',
    #                                 evi_filepath=
    #                                 'pickleJar/withoutAE-524/prototypeGradient/FN4_proto%d_nu%s' % (num_proto,str(nu)),
    #                                 mid_filepath=
    #                                 'pickleJar/withoutAE-524/prototypeGradient/mid/FN4_proto%d_nu%s' % (num_proto,str(nu)),
    #                                 load_weights=0, train_DS_Layers=0, plot_DS_layer_loss=0,
    #                                 x_train=x_resampled_2D, y_train=y_resampled_onehot,
    #                                 x_test=old_test_2D, y_test=old_test_label,
    #                                 is_load=0,output_confusion_matrix=1)

    """ set-valued mission """
    num_class = 5
    class_set = list(range(num_class))
    act_set = sv.PowerSets(class_set, no_empty=True, is_sorted=True)
    UM = sv.utility_mtx(num_class, act_set=act_set, class_set=class_set, tol_i=3, m=0)
    sv.set_valued_evidential_FitNet4(num_class=5, number_act_set=31,
                                    filepath='E:/Obj/pickleJar/NoNovelty/EVI_FN4_NSL_proto100_nu0.1',
                                    nu=0.1, prototypes=100, tol=0.8,
                                    utility_matrix=UM, is_load=True, act_set=act_set,
                                    x_test=test_2D, numerical_y_test=old_test_label_numerical)

