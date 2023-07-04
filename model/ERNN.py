import matplotlib.pyplot as plt
from math import sqrt
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd
import numpy as np
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras import layers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


def vanilla_RNN(data_dimension,hidden_units,num_class,
                model_filepath,is_train, is_load,
                x_train,y_train,x_test,y_test,
                pic_filepath_loss,pic_filepath_acc,
                output_confusion_matrix):

    inputs = layers.Input((1,data_dimension))
    r1 = layers.SimpleRNN(units=hidden_units,return_sequences=True,dropout=0.5)(inputs)
    flatten1 = layers.Flatten()(r1)
    outputs = layers.Dense(num_class, activation='softmax')(flatten1)

    model = keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=keras.optimizers.Nadam(
        learning_rate=0.001,beta_1=0.9, beta_2=0.999,epsilon=None,schedule_decay=0.004),
        loss='CategoricalCrossentropy',metrics=['accuracy'])
    model.summary()
    checkpoint_callback = ModelCheckpoint(model_filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                          save_weights_only=True, save_frequency=1)
    if is_train == 1:
        h = model.fit(x_train, y_train, batch_size=32, epochs=60,
                      verbose=1, callbacks=[checkpoint_callback], validation_data=(x_test, y_test), shuffle=True)

        history = h.history
        epochs = range(len(history['loss']))
        plt.plot(epochs, history['loss'], 'royalblue', label='Train loss')
        plt.plot(epochs, history['val_loss'], 'crimson', label='Valid loss')  # Test loss
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.rcParams["figure.dpi"] = 300
        plt.legend()
        plt.savefig(pic_filepath_loss, dpi=300)
        # plt.show()
        plt.close()

        plt.plot(epochs, history['accuracy'], 'royalblue', label='Train accuracy')
        plt.plot(epochs, history['val_accuracy'], 'crimson', label='Valid accuracy')  # Test loss
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.rcParams["figure.dpi"] = 300
        plt.legend()
        plt.savefig(pic_filepath_acc, dpi=300)
        # plt.show()
        plt.close()
    if is_load == 1:
        model.load_weights(model_filepath).expect_partial()
        if output_confusion_matrix == 1:
            y_pred = model.predict(x_test)
            y_pred = y_pred.argmax(axis=1)
            y_test = y_test.argmax(axis=1)
            confusion_mtx = confusion_matrix(y_test, y_pred)
            normalized_confusion_mtx = np.round(confusion_matrix(y_test, y_pred, normalize='true'), 3)
            f1 = f1_score(y_true=y_test, y_pred=y_pred, labels=None, average='macro')
            acc = accuracy_score(y_true=y_test, y_pred=y_pred)
            recall = recall_score(y_true=y_test, y_pred=y_pred, labels=None, average='macro')
            preccision = precision_score(y_true=y_test, y_pred=y_pred, labels=None, average='macro')
            print(confusion_mtx)
            print(normalized_confusion_mtx)
            print('accuracy: ' + str(acc))
            print('recall: ' + str(recall))
            print('precision: ' + str(preccision))
            print('F1-score: ' + str(f1))

