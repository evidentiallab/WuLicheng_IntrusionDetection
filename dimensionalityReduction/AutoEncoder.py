import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Dense, Input, Activation
from keras.models import load_model, Model
import matplotlib.pyplot as plt
import time


# tf.keras.layers.LeakyReLU(alpha=0.01)
def train_AE_123_100(train,test, is_train, is_polt):
    # 定义自编码器模型
    input_data = Input(shape=(123,))
    hidden_layer = Dense(100, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(input_data)
    output_data = Dense(123, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(hidden_layer)
    autoencoder = Model(inputs=input_data, outputs=output_data)
    encoder = Model(inputs=input_data, outputs=hidden_layer)
    # autoencoder.summary()
    if is_train == 1:
        # 训练自编码器模型
        autoencoder.compile(optimizer='adam', loss='mse')
        h = autoencoder.fit(train, train, epochs=20, batch_size=128)
        autoencoder.save('AE123-100.h5')
        # encoder.save('ENC123-100.h5')
        if is_polt == 1:
            history = h.history
            epochs = range(len(history['loss']))
            plt.plot(epochs, history['loss'])
            plt.xlabel("Epochs")
            plt.ylabel("Reconstruction Error")
            plt.rcParams["figure.dpi"] = 300
            plt.savefig('../pic/123-100(leaky-leaky).png', dpi=300)
            plt.show()

    if is_train == 0:
        # 使用自编码器模型进行预测
        # AE = load_model('AE123-100.h5')
        # 生成训练集
        enc_data = encoder.predict(train)
        enc_data = np.around(enc_data,5)
        pd.DataFrame(enc_data).to_csv('../dataset/KDDCUP99/Encoded/train(123_100).csv',header=False, index=False)
        print("shape of enc_train:" + str(enc_data.shape))
        # 生成测试集
        enc_test = encoder.predict(test)
        enc_test = np.around(enc_test, 5)
        pd.DataFrame(enc_test).to_csv('../dataset/KDDCUP99/Encoded/test(123_100).csv', header=False, index=False)
        # dec_data = AE.predict(data)
        # dec_data = np.around(dec_data,5)
        # pd.DataFrame(dec_data).to_csv('../dataset/KDDCUP99/Encoded/leaky-leaky.csv',header=False, index=False)
        print("shape of enc_test:" + str(enc_test.shape))
        # return enc_data,enc_test


def train_AE_123_100_81(train, test, is_train, is_polt):
    # 定义自编码器模型
    input_data = Input(shape=(123,))
    hidden_layer0 = Dense(100, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(input_data)
    target_layer = Dense(81, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(hidden_layer0)
    hidden_layer1 = Dense(100, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(target_layer)
    output_data = Dense(123, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(hidden_layer1)
    autoencoder = Model(inputs=input_data, outputs=output_data)
    encoder = Model(inputs=input_data, outputs=target_layer)
    # autoencoder.summary()
    if is_train == 1:
        # 训练自编码器模型
        autoencoder.compile(optimizer='adam', loss='mse')
        h = autoencoder.fit(train, train, epochs=20, batch_size=128)
        autoencoder.save('AE123-100-81.h5')
        # encoder.save('ENC123-100.h5')
        if is_polt == 1:
            history = h.history
            epochs = range(len(history['loss']))
            plt.plot(epochs, history['loss'])
            plt.xlabel("Epochs")
            plt.ylabel("Reconstruction Error")
            plt.rcParams["figure.dpi"] = 300
            plt.savefig('../pic/123-100-81(leaky all).png', dpi=300)
            plt.show()

    if is_train == 0:
        # 使用自编码器模型进行预测
        # AE = load_model('AE123-100.h5')
        # 生成训练集
        enc_data = encoder.predict(train)
        enc_data = np.around(enc_data, 5)
        pd.DataFrame(enc_data).to_csv('../dataset/KDDCUP99/Encoded/train(123_100_81).csv', header=False, index=False)
        print("shape of enc_train:" + str(enc_data.shape))
        # 生成测试集
        enc_test = encoder.predict(test)
        enc_test = np.around(enc_test, 5)
        pd.DataFrame(enc_test).to_csv('../dataset/KDDCUP99/Encoded/test(123_100_81).csv', header=False, index=False)
        # dec_data = AE.predict(data)
        # dec_data = np.around(dec_data,5)
        # pd.DataFrame(dec_data).to_csv('../dataset/KDDCUP99/Encoded/leaky-leaky.csv',header=False, index=False)
        print("shape of enc_test:" + str(enc_test.shape))
        # return enc_data,enc_test


def train_AE_123_100_64(train, test, is_train, is_polt):
    # 定义自编码器模型
    input_data = Input(shape=(123,))
    hidden_layer0 = Dense(100, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(input_data)
    target_layer = Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(hidden_layer0)
    hidden_layer1 = Dense(100, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(target_layer)
    output_data = Dense(123, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(hidden_layer1)
    autoencoder = Model(inputs=input_data, outputs=output_data)
    encoder = Model(inputs=input_data, outputs=target_layer)
    autoencoder.summary()
    if is_train == 1:
        # 训练自编码器模型
        autoencoder.compile(optimizer='adam', loss='mse')
        h = autoencoder.fit(train, train, epochs=20, batch_size=128)
        autoencoder.save('AE123-100-64.h5')
        # encoder.save('ENC123-100.h5')
        if is_polt == 1:
            history = h.history
            epochs = range(len(history['loss']))
            plt.plot(epochs, history['loss'])
            plt.xlabel("Epochs")
            plt.ylabel("Reconstruction Error")
            plt.rcParams["figure.dpi"] = 300
            plt.savefig('../pic/123-100-64(leaky all).png', dpi=300)
            plt.show()

    if is_train == 0:
        # 使用自编码器模型进行预测
        # AE = load_model('AE123-100.h5')
        # 生成训练集
        enc_data = encoder.predict(train)
        enc_data = np.around(enc_data, 5)
        pd.DataFrame(enc_data).to_csv('../dataset/KDDCUP99/Encoded/train(123_100_64).csv', header=False, index=False)
        print("shape of enc_train:" + str(enc_data.shape))
        # 生成测试集
        enc_test = encoder.predict(test)
        enc_test = np.around(enc_test, 5)
        pd.DataFrame(enc_test).to_csv('../dataset/KDDCUP99/Encoded/test(123_100_64).csv', header=False, index=False)
        # dec_data = AE.predict(data)
        # dec_data = np.around(dec_data,5)
        # pd.DataFrame(dec_data).to_csv('../dataset/KDDCUP99/Encoded/leaky-leaky.csv',header=False, index=False)
        print("shape of enc_test:" + str(enc_test.shape))
        # return enc_data,enc_test


if __name__ == '__main__':
    # 加载训练集
    df = pd.read_csv('../dataset/KDDCUP99/concatenated/train-MinMax.csv',header=0)
    # df.fillna(0, inplace=True)
    train = df.values
    # 加载测试集
    df2 = pd.read_csv('../dataset/KDDCUP99/concatenated/test-MinMax.csv',header=None)
    test = df2.values
    # 训练or加载AE
    # train_AE_135_100(train,test,is_train=0,is_polt=1)
    # train_AE_123_100_81(train, test, is_train=0, is_polt=1)
    train_AE_123_100_64(train, test, is_train=0, is_polt=1)



