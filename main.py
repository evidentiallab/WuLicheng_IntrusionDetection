import pandas as pd
import matplotlib.pyplot as plt
from model import ECNN_NSL
import model.Set_valued_ECNN as sv

if __name__ == '__main__':
    smote_train_filepath = 'D:/IDdataset/processedNSL/SMOTE/NSL_train_smote.csv'
    smote_train_label_filepath = 'D:/IDdataset/processedNSL/SMOTE/NSL_train_label_smote_onehot.csv'

    train_filepath = 'D:/IDdataset/processedNSL/train(129-100).csv'
    test_filepath = 'D:/IDdataset/processedNSL/test(129-100).csv'

    train_label_filepath = 'D:/IDdataset/train_label.csv'
    test_label_filepath = 'D:/IDdataset/test_label.csv'

    """normal train"""
    train = pd.read_csv(train_filepath, header=None).values
    train_2D = train.reshape(train.shape[0], 10, 10)
    train_label = pd.read_csv(train_label_filepath, header=0)
    # print(train_2D.shape)
    # print(train_label.value_counts())
    train_label = train_label.values

    """SMOTE train"""
    SMOTE_train = pd.read_csv(smote_train_filepath, header=None).values
    SMOTE_train_2D = SMOTE_train.reshape(SMOTE_train.shape[0], 10, 10)
    SMOTE_label = pd.read_csv(smote_train_label_filepath, header=0)
    # print(SMOTE_train_2D.shape)
    # print(SMOTE_label.value_counts())
    SMOTE_label = SMOTE_label.values

    """normal test"""
    test = pd.read_csv(test_filepath, header=None).values
    test_2D = test.reshape(test.shape[0], 10, 10)
    test_label = pd.read_csv(test_label_filepath, header=0)
    # print(test_2D.shape)
    # print(test_label.value_counts())
    test_label = test_label.values

    """no novelty test"""
    no_novelty_test = pd.read_csv('D:/IDdataset/processedNSL/test(129-100)_no_novelty.csv',header=None).values
    no_novelty_test_2D = no_novelty_test.reshape(no_novelty_test.shape[0], 10, 10)
    no_novelty_test_label = pd.read_csv('D:/IDdataset/no_novelty_test_label.csv',header=0)
    # print(no_novelty_test_2D.shape)
    # print(no_novelty_test_label.value_counts())
    no_novelty_test_label = no_novelty_test_label.values

    no_novelty_test_label_numerical = pd.read_csv('D:/IDdataset/no_novelty_test_label_numerical.csv',header=0).values
    # ECNN_NSL.probabilistic_FitNet4_simplify(data_WIDTH=10, data_HEIGHT=10, num_class=5,
    #                                         is_train=1, is_load=1, model_filepath='pickleJar/NoNovelty/PR_FN4S_NSL',
    #                                         pic_filepath_loss='pic/PR-FN4S-no-noveltyNSL-loss',pic_filepath_acc='pic/PR-FN4S-no-noveltyNSL-acc',
    #                                         x_train=train_2D, y_train=train_label, x_test=no_novelty_test_2D, y_test=no_novelty_test_label,
    #                                         output_confusion_matrix=1)
    # ECNN_NSL.probabilistic_FitNet4(data_WIDTH=10, data_HEIGHT=10, num_class=5,
    #                                is_train=0, is_load=1, model_filepath='pickleJar/NoNovelty/PR_FN4_NSL',
    #                                pic_filepath_loss='pic/PR-FN4-no-noveltyNSL-loss',
    #                                pic_filepath_acc='pic/PR-FN4-no-noveltyNSL-acc',
    #                                x_train=train_2D, y_train=train_label,
    #                                x_test=no_novelty_test_2D,y_test=no_novelty_test_label,
    #                                output_confusion_matrix=1)

    # SMOTE-Train
    # ECNN_NSL.probabilistic_FitNet4_simplify(data_WIDTH=10, data_HEIGHT=10, num_class=5,
    #                                is_train=1, is_load=1, model_filepath='pickleJar/NoNovelty/PR_FN4S_SMOTE_NSL',
    #                                pic_filepath_loss='pic/PR-FN4-no-novelty-SMOTE-NSL-loss',
    #                                pic_filepath_acc='pic/PR-FN4-no-noveltyNSL-acc',
    #                                x_train=SMOTE_train_2D, y_train=SMOTE_label,
    #                                x_test=no_novelty_test_2D,y_test=no_novelty_test_label,
    #                                output_confusion_matrix=1)
    #
    # ECNN_NSL.probabilistic_FitNet4(data_WIDTH=10, data_HEIGHT=10, num_class=5,
    #                                is_train=0, is_load=1, model_filepath='pickleJar/NoNovelty/PR_FN4_SMOTE_NSL',
    #                                pic_filepath_loss='pic/PR-FN4-no-novelty-SMOTE-NSL-loss',
    #                                pic_filepath_acc='pic/PR-FN4-no-noveltyNSL-acc',
    #                                x_train=SMOTE_train_2D, y_train=SMOTE_label,
    #                                x_test=no_novelty_test_2D,y_test=no_novelty_test_label,
    #                                output_confusion_matrix=1)

    ECNN_NSL.evidential_FitNet4(data_WIDTH=10, data_HEIGHT=10, num_class=5, prototypes=100, nu=0.2,
                                model_filepath='pickleJar/NoNovelty/PR_FN4_NSL', load_weights=1,
                                train_DS_Layers=1, plot_DS_layer_loss=0,
                                evi_filepath='pickleJar/NoNovelty/EVI_FN4_NSL_proto100_nu0.2',
                                mid_filepath='pickleJar/mid/FN4_NSL_proto100_nu0.2',
                                x_train=train_2D, y_train=train_label,
                                x_test=no_novelty_test_2D, y_test=no_novelty_test_label,
                                is_load=0, output_confusion_matrix=1)

    ECNN_NSL.evidential_FitNet4(data_WIDTH=10, data_HEIGHT=10, num_class=5, prototypes=100, nu=0.3,
                                model_filepath='pickleJar/NoNovelty/PR_FN4_NSL', load_weights=1,
                                train_DS_Layers=1, plot_DS_layer_loss=0,
                                evi_filepath='pickleJar/NoNovelty/EVI_FN4_NSL_proto100_nu0.3',
                                mid_filepath='pickleJar/mid/FN4_NSL_proto100_nu0.3',
                                x_train=train_2D, y_train=train_label,
                                x_test=no_novelty_test_2D, y_test=no_novelty_test_label,
                                is_load=0, output_confusion_matrix=1)

    ECNN_NSL.evidential_FitNet4(data_WIDTH=10, data_HEIGHT=10, num_class=5, prototypes=100, nu=0.4,
                                model_filepath='pickleJar/NoNovelty/PR_FN4_NSL', load_weights=1,
                                train_DS_Layers=1, plot_DS_layer_loss=0,
                                evi_filepath='pickleJar/NoNovelty/EVI_FN4_NSL_proto100_nu0.4',
                                mid_filepath='pickleJar/mid/FN4_NSL_proto100_nu0.4',
                                x_train=train_2D, y_train=train_label,
                                x_test=no_novelty_test_2D, y_test=no_novelty_test_label,
                                is_load=0, output_confusion_matrix=1)

    ECNN_NSL.evidential_FitNet4(data_WIDTH=10, data_HEIGHT=10, num_class=5, prototypes=100, nu=0.5,
                                model_filepath='pickleJar/NoNovelty/PR_FN4_NSL', load_weights=1,
                                train_DS_Layers=1, plot_DS_layer_loss=0,
                                evi_filepath='pickleJar/NoNovelty/EVI_FN4_NSL_proto100_nu0.5',
                                mid_filepath='pickleJar/mid/FN4_NSL_proto100_nu0.5',
                                x_train=train_2D, y_train=train_label,
                                x_test=no_novelty_test_2D, y_test=no_novelty_test_label,
                                is_load=0, output_confusion_matrix=1)

    ECNN_NSL.evidential_FitNet4(data_WIDTH=10, data_HEIGHT=10, num_class=5, prototypes=100, nu=0.6,
                                model_filepath='pickleJar/NoNovelty/PR_FN4_NSL', load_weights=1,
                                train_DS_Layers=1, plot_DS_layer_loss=0,
                                evi_filepath='pickleJar/NoNovelty/EVI_FN4_NSL_proto100_nu0.6',
                                mid_filepath='pickleJar/mid/FN4_NSL_proto100_nu0.6',
                                x_train=train_2D, y_train=train_label,
                                x_test=no_novelty_test_2D, y_test=no_novelty_test_label,
                                is_load=0, output_confusion_matrix=1)

    ECNN_NSL.evidential_FitNet4(data_WIDTH=10, data_HEIGHT=10, num_class=5, prototypes=100, nu=0.7,
                                model_filepath='pickleJar/NoNovelty/PR_FN4_NSL', load_weights=1,
                                train_DS_Layers=1, plot_DS_layer_loss=0,
                                evi_filepath='pickleJar/NoNovelty/EVI_FN4_NSL_proto100_nu0.7',
                                mid_filepath='pickleJar/mid/FN4_NSL_proto100_nu0.7',
                                x_train=train_2D, y_train=train_label,
                                x_test=no_novelty_test_2D, y_test=no_novelty_test_label,
                                is_load=0, output_confusion_matrix=1)

    ECNN_NSL.evidential_FitNet4(data_WIDTH=10, data_HEIGHT=10, num_class=5, prototypes=100, nu=0.8,
                                model_filepath='pickleJar/NoNovelty/PR_FN4_NSL', load_weights=1,
                                train_DS_Layers=1, plot_DS_layer_loss=0,
                                evi_filepath='pickleJar/NoNovelty/EVI_FN4_NSL_proto100_nu0.8',
                                mid_filepath='pickleJar/mid/FN4_NSL_proto100_nu0.8',
                                x_train=train_2D, y_train=train_label,
                                x_test=no_novelty_test_2D, y_test=no_novelty_test_label,
                                is_load=0, output_confusion_matrix=1)

    ECNN_NSL.evidential_FitNet4(data_WIDTH=10, data_HEIGHT=10, num_class=5, prototypes=100, nu=0.9,
                                model_filepath='pickleJar/NoNovelty/PR_FN4_NSL', load_weights=1,
                                train_DS_Layers=1, plot_DS_layer_loss=0,
                                evi_filepath='pickleJar/NoNovelty/EVI_FN4_NSL_proto100_nu0.9',
                                mid_filepath='pickleJar/mid/FN4_NSL_proto100_nu0.9',
                                x_train=train_2D, y_train=train_label,
                                x_test=no_novelty_test_2D, y_test=no_novelty_test_label,
                                is_load=0, output_confusion_matrix=1)

    ECNN_NSL.evidential_FitNet4(data_WIDTH=10, data_HEIGHT=10, num_class=5, prototypes=100, nu=1,
                                model_filepath='pickleJar/NoNovelty/PR_FN4_NSL', load_weights=1,
                                train_DS_Layers=1, plot_DS_layer_loss=0,
                                evi_filepath='pickleJar/NoNovelty/EVI_FN4_NSL_proto100_nu1.0',
                                mid_filepath='pickleJar/mid/FN4_NSL_proto100_nu1.0',
                                x_train=train_2D, y_train=train_label,
                                x_test=no_novelty_test_2D, y_test=no_novelty_test_label,
                                is_load=0, output_confusion_matrix=1)

    ECNN_NSL.evidential_FitNet4(data_WIDTH=10, data_HEIGHT=10, num_class=5, prototypes=100, nu=0,
                                model_filepath='pickleJar/NoNovelty/PR_FN4_NSL', load_weights=1,
                                train_DS_Layers=1, plot_DS_layer_loss=0,
                                evi_filepath='pickleJar/NoNovelty/EVI_FN4_NSL_proto100_nu0',
                                mid_filepath='pickleJar/mid/FN4_NSL_proto100_nu0',
                                x_train=train_2D, y_train=train_label,
                                x_test=no_novelty_test_2D, y_test=no_novelty_test_label,
                                is_load=0, output_confusion_matrix=1)
    #
    # num_class = 5
    # class_set = list(range(num_class))
    # act_set = sv.PowerSets(class_set, no_empty=True, is_sorted=True)
    # # print(len(act_set))
    # UM = sv.utility_mtx(num_class, act_set=act_set, class_set=class_set)
    # print(UM.shape)
    # sv.set_valued_evidential_FitNet4(num_class=5,number_act_set=31,filepath='pickleJar/NoNovelty/EVI_FN4_NSL_proto40_nu0.1',
    #                                 nu=0.1,prototypes=40,tol=0.9,utility_matrix=UM,is_load=True,act_set=act_set,
    #                                 x_test=no_novelty_test_2D, numerical_y_test=no_novelty_test_label_numerical)






