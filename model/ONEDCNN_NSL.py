import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from libs import ds_layer               # Dempster-Shafer layer
from libs import utility_layer_train    # Utility layer for training
import pandas as pd
import matplotlib.pyplot as plt
import ECNN_NSL


def probabilistic_FN4S_Conv1D(data_num,data_WIDTH, num_class,
                              is_train, is_load, model_filepath,
                              x_train, y_train, x_test, y_test,
                              pic_fp0,pic_fp1,
                              output_confusion_matrix):
    inputs = tf.keras.layers.Input((data_WIDTH,1))

    # c1_1 = keras.layers.Conv1D(32,(1, 3),kernel_initializer='he_normal',padding='same')(inputs)
    c1_1 = keras.layers.Conv1D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1_2 = keras.layers.Conv1D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c1_1)
    c1_3 = keras.layers.Conv1D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c1_2)
    c1_4 = keras.layers.Conv1D(48, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c1_3)
    c1_5 = keras.layers.Conv1D(48, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c1_4)
    bt1 = keras.layers.BatchNormalization()(c1_5)
    p1 = keras.layers.MaxPooling1D(3)(bt1)
    dr1 = keras.layers.Dropout(0.5)(p1)

    c2_1 = tf.keras.layers.Conv1D(80, 3, activation='relu', kernel_initializer='he_normal', padding='same')(dr1)
    c2_2 = tf.keras.layers.Conv1D(80, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c2_1)
    c2_3 = tf.keras.layers.Conv1D(80, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c2_2)
    c2_4 = tf.keras.layers.Conv1D(80, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c2_3)
    c2_5 = tf.keras.layers.Conv1D(80, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c2_4)
    bt2 = tf.keras.layers.BatchNormalization()(c2_5)
    p2 = tf.keras.layers.MaxPooling1D(2)(bt2)
    dr2 = tf.keras.layers.Dropout(0.5)(p2)

    flatten1 = tf.keras.layers.Flatten()(dr2)

    outputs = tf.keras.layers.Dense(num_class, activation='softmax')(flatten1)
    model_PR = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model_PR.compile(optimizer=keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
                                                      schedule_decay=0.004), loss='CategoricalCrossentropy',metrics=['accuracy'])
    model_PR.summary()

    checkpoint_callback = ModelCheckpoint(model_filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                          save_weights_only=True, save_frequency=1)

    if is_train == 1:
        h = model_PR.fit(x_train, y_train, batch_size=32, epochs=40,
                 verbose=1, callbacks=[checkpoint_callback], validation_data=(x_test, y_test), shuffle=True)

        history = h.history
        epochs = range(len(history['loss']))
        plt.figure()
        plt.plot(epochs, history['loss'],'b',label='Train loss')
        plt.plot(epochs, history['val_loss'],'r',label='Valid loss')    # Test loss
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.rcParams["figure.dpi"] = 300
        plt.legend()
        plt.savefig(pic_fp0, dpi=300)
        # plt.show()
        plt.close()

        plt.figure()
        plt.plot(epochs, history['accuracy'],'b', label='Train accuracy')
        plt.plot(epochs, history['val_accuracy'],'r', label='Valid accuracy')   # Test loss
        plt.xlabel("Epochs")
        plt.ylabel("accuracy")
        plt.rcParams["figure.dpi"] = 300
        plt.legend()
        plt.savefig(pic_fp1, dpi=300)
        # plt.show()
        plt.close()

    if is_load == 1:
        model_PR.load_weights(model_filepath).expect_partial()
        # model_PR.load_weights(filepath1)
        # model_PR.evaluate(x_train, y_train, batch_size=25, verbose=1)
        # model_PR.evaluate(x_test, y_test, batch_size=32, verbose=1)
        if output_confusion_matrix == 1:
            y_pred = model_PR.predict(x_test)
            y_pred = y_pred.argmax(axis=1)
            y_test = y_test.argmax(axis=1)
            # print(y_pred)
            # print(y_test)
            # print(type(y_pred))
            # print(type(y_test))
            # print(y_test.shape)
            # print(y_pred.shape)
            confusion_mtx = confusion_matrix(y_test,y_pred)
            f1 = f1_score(y_true=y_test, y_pred=y_pred, labels=None, average='macro')
            acc = accuracy_score(y_true=y_test, y_pred=y_pred)
            recall = recall_score(y_true=y_test, y_pred=y_pred, labels=None, average='macro')
            preccision = precision_score(y_true=y_test, y_pred=y_pred, labels=None, average='macro')
            print(confusion_mtx)
            print('accuracy: ' + str(acc))
            print('recall: ' + str(recall))
            print('precision: ' + str(preccision))
            print('F1-score: ' + str(f1))


def evidential_FN4S_Conv1D(data_WIDTH, num_class,prototypes,is_load,
                           model_filepath,DS_filepath,load_weights, train_DS_Layers,
                           x_train, y_train, x_test, y_test,
                           output_confusion_matrix):
    inputs = tf.keras.layers.Input((data_WIDTH,1))

    # c1_1 = keras.layers.Conv1D(32,(1, 3),kernel_initializer='he_normal',padding='same')(inputs)
    c1_1 = keras.layers.Conv1D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1_2 = keras.layers.Conv1D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c1_1)
    c1_3 = keras.layers.Conv1D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c1_2)
    c1_4 = keras.layers.Conv1D(48, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c1_3)
    c1_5 = keras.layers.Conv1D(48, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c1_4)
    bt1 = keras.layers.BatchNormalization()(c1_5)
    p1 = keras.layers.MaxPooling1D(3)(bt1)
    dr1 = keras.layers.Dropout(0.5)(p1)

    c2_1 = tf.keras.layers.Conv1D(80, 3, activation='relu', kernel_initializer='he_normal', padding='same')(dr1)
    c2_2 = tf.keras.layers.Conv1D(80, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c2_1)
    c2_3 = tf.keras.layers.Conv1D(80, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c2_2)
    c2_4 = tf.keras.layers.Conv1D(80, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c2_3)
    c2_5 = tf.keras.layers.Conv1D(80, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c2_4)
    bt2 = tf.keras.layers.BatchNormalization()(c2_5)
    p2 = tf.keras.layers.MaxPooling1D(2)(bt2)
    dr2 = tf.keras.layers.Dropout(0.5)(p2)

    flatten1 = tf.keras.layers.Flatten()(dr2)

    # DS layer
    ED = ds_layer.DS1(prototypes, 1680)(flatten1)
    ED_ac = ds_layer.DS1_activate(prototypes)(ED)
    mass_prototypes = ds_layer.DS2(prototypes, num_class)(ED_ac)
    mass_prototypes_omega = ds_layer.DS2_omega(prototypes, num_class)(mass_prototypes)
    mass_Dempster = ds_layer.DS3_Dempster(prototypes, num_class)(mass_prototypes_omega)
    mass_Dempster_normalize = ds_layer.DS3_normalize()(mass_Dempster)

    # Utility layer for training
    outputs = utility_layer_train.DM(0.9, num_class)(mass_Dempster_normalize)
    model_evi = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model_evi.compile(
        optimizer=keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
                                         schedule_decay=0.004),
        loss='CategoricalCrossentropy',
        metrics=['accuracy'])
    # model_evi.summary()

    if load_weights == 1:
        # load the weights of probabilistic classifier
        # and get the feature for training DS layers
        model_evi.load_weights(model_filepath).expect_partial()
        feature = tf.keras.Model(inputs=[inputs],outputs=[flatten1])
        train_feature_for_DS = feature.predict(x_train)
        test_feature_for_DS = feature.predict(x_test)

    if train_DS_Layers == 1:
        # training DS layers
        inputss = tf.keras.layers.Input(1680)
        ED = ds_layer.DS1(prototypes, 1680)(inputss)
        ED_ac = ds_layer.DS1_activate(prototypes)(ED)
        mass_prototypes = ds_layer.DS2(prototypes, num_class)(ED_ac)
        mass_prototypes_omega = ds_layer.DS2_omega(prototypes, num_class)(mass_prototypes)
        mass_Dempster = ds_layer.DS3_Dempster(prototypes, num_class)(mass_prototypes_omega)
        mass_Dempster_normalize = ds_layer.DS3_normalize()(mass_Dempster)
        outputss = utility_layer_train.DM(0.9, num_class)(mass_Dempster_normalize)
        model_mid = tf.keras.Model(inputs=[inputss], outputs=[outputss])
        model_mid.compile(
            optimizer=keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004),
            # 0.001
            loss='CategoricalCrossentropy',
            metrics=['accuracy'])
        # model_mid.summary()
        h = model_mid.fit(train_feature_for_DS, y_train, batch_size=64, epochs=2, verbose=1,
                      validation_data=(test_feature_for_DS, y_test), shuffle=True)

        # feed the trained paramters to the evidential model
        model_evi.load_weights(model_filepath).expect_partial()
        DS1_W = tf.reshape(model_mid.layers[1].get_weights()[0], [prototypes, 1680])
        DS1_activate_W = model_mid.layers[2].get_weights()
        DS2_W = model_mid.layers[3].get_weights()
        model_evi.layers[26].set_weights(DS1_W)
        model_evi.layers[27].set_weights(DS1_activate_W)
        model_evi.layers[28].set_weights(DS2_W)

        checkpoint_callback = ModelCheckpoint(
            DS_filepath, monitor='val_accuracy', verbose=1,
            save_best_only=True, save_weights_only=True,
            save_frequency=1)
        model_evi.fit(x_train, y_train,
                      batch_size=64, epochs=4, verbose=1,
                      callbacks=[checkpoint_callback], validation_data=(x_test, y_test), shuffle=True)

    if is_load == 1:
        model_evi.load_weights(DS_filepath).expect_partial()
        model_evi.evaluate(x_train, y_train, batch_size=64, verbose=1)
        model_evi.evaluate(x_test, y_test, batch_size=64, verbose=1)
        if output_confusion_matrix == 1:
            y_pred = model_evi.predict(x_test)
            y_pred = y_pred.argmax(axis=1)
            y_test = y_test.argmax(axis=1)
            confusion_mtx = confusion_matrix(y_test, y_pred)
            f1 = f1_score(y_true=y_test, y_pred=y_pred, labels=None, average='macro')
            acc = accuracy_score(y_true=y_test, y_pred=y_pred)
            recall = recall_score(y_true=y_test, y_pred=y_pred, labels=None, average='macro')
            preccision = precision_score(y_true=y_test, y_pred=y_pred, labels=None, average='macro')
            print(confusion_mtx)
            print('accuracy: ' + str(acc))
            print('recall: ' + str(recall))
            print('precision: ' + str(preccision))
            print('F1-score: ' + str(f1))


if __name__ == '__main__':
    train_fp = 'D:/IDdataset/processedNSL/train(MinMax).csv'
    train_label_fp = 'D:/IDdataset/train_label.csv'
    test_fp = 'D:/IDdataset/processedNSL/test(MinMax).csv'
    test_label_fp = 'D:/IDdataset/test_label.csv'

    smote_fp = 'D:/IDdataset/processedNSL/SMOTE/NSL_train_MinMax.csv'
    smote_label_fp = 'D:/IDdataset/processedNSL/SMOTE/NSL_train_MinMax_label_onehot.csv'

    model_fp_0 = 'pickleJar/Conv1D/PR_FN4S_NSL'
    model_fp_1 = 'pickleJar/Conv1D/PR_FN4S_NSL_SMOTE'

    pic_fp_00 = 'pic/PR-FN4S1D(NSL)-loss.png'
    pic_fp_01 = 'pic/PR-FN4S1D(NSL)-accuracy.png'
    pic_fp_10 = 'pic/PR-FN4S1D(NSL-SMOTE)-loss.png'
    pic_fp_11 = 'pic/PR-FN4S1D(NSL-SMOTE)-accuracy.png'

    DS_filepath = 'pickleJar/Conv1D/EVI_FN4S_NSL'
    DS_filepath_smote = 'pickleJar/Conv1D/EVI_FN4S_NSL_SMOTE'

    df_train = pd.read_csv(train_fp,header=0)
    df_train_label = pd.read_csv(train_label_fp,header=0)
    train = df_train.values
    train_label = df_train_label.values
    # print(train.shape[0])
    # print(train.shape[1])
    df_test = pd.read_csv(test_fp, header=0)
    df_test_label = pd.read_csv(test_label_fp, header=0)
    print(df_test_label.value_counts())
    test = df_test.values
    test_label = df_test_label.values

    df_smote = pd.read_csv(smote_fp,header=0)
    df_smote_label = pd.read_csv(smote_label_fp,header=0)
    print(df_smote_label.value_counts())
    smote = df_smote.values
    smote_label = df_smote_label.values

    # 2D dataset
    smote_2D = pd.read_csv('D:/IDdataset/processedNSL/SMOTE/NSL_train_smote.csv',header=None)
    smote_2D = smote_2D.values
    smote_2D = smote_2D.reshape(smote_2D.shape[0], 10, 10)
    print(smote_2D.shape)
    smote_2D_label = pd.read_csv('D:/IDdataset/processedNSL/SMOTE/NSL_train_label_smote_onehot.csv',header=0)
    smote_2D_label = smote_2D_label.values
    print(smote_2D_label.shape)
    test_2D = pd.read_csv('D:/IDdataset/processedNSL/test(129-100).csv',header=None)
    test_2D = test_2D.values.reshape(test_2D.shape[0], 10, 10)
    test_2D_label = pd.read_csv('D:/IDdataset/test_label.csv',header=0)
    test_2D_label = test_2D_label.values
    print(test_2D.shape)
    print(test_2D_label.shape)

    # probabilistic_FN4S_Conv1D(data_num=train.shape[0], data_WIDTH=train.shape[1], num_class=5,
    #                           is_train=1, is_load=1, model_filepath=model_fp_0,
    #                           pic_fp0=pic_fp_00,pic_fp1=pic_fp_01,
    #                           x_train=train, y_train=train_label, x_test=test, y_test=test_label,
    #                           output_confusion_matrix=1)

    # probabilistic_FN4S_Conv1D(data_num=smote.shape[0], data_WIDTH=smote.shape[1], num_class=5,
    #                           is_train=1, is_load=1, model_filepath=model_fp_1,
    #                           pic_fp0=pic_fp_10, pic_fp1=pic_fp_11,
    #                           x_train=smote, y_train=smote_label, x_test=test, y_test=test_label,
    #                           output_confusion_matrix=1)

    # evidential_FN4S_Conv1D(data_WIDTH=train.shape[1],num_class=5,prototypes=20,
    #                        model_filepath=model_fp_0,DS_filepath=DS_filepath,
    #                        load_weights=1,train_DS_Layers=1,
    #                        x_train=train, y_train=train_label, x_test=test, y_test=test_label,
    #                        is_load=0,output_confusion_matrix=0)
    '''Train FN4S with SMOTE NSL'''
    model_fp = 'pickleJar/Probabilistic/PR_FN4S_SMOTE_NSL'
    # ECNN_NSL.probabilistic_FitNet4_simplify(data_WIDTH=10, data_HEIGHT=10, num_class=5,
    #                       is_train=1, is_load=1, output_confusion_matrix=1,
    #                       x_train=smote_2D, y_train=smote_2D_label, model_filepath=model_fp,
    #                       x_test=test_2D, y_test=test_2D_label)

    ECNN_NSL.evidential_FitNet4_simplify(data_WIDTH=10, data_HEIGHT=10, num_class=5, prototypes=20, nu=1, is_load=1,
                                         model_filepath=model_fp, DS_filepath='pickleJar/Evidential/EVI_FN4S_SMOTE_NSL_100proto_nu1',
                                         load_weights=1, train_DS_Layers=1, plot_DS_layer_loss=0,
                                         x_train=smote_2D, y_train=smote_2D_label,
                                         x_test=test_2D, y_test=test_2D_label,
                                         output_confusion_matrix=1)

