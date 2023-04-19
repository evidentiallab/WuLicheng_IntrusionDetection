import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Dense, Input, Activation
from keras.models import load_model, Model
import matplotlib.pyplot as plt

train_filepath = 'D:/IDdataset/processedNSL/train(MinMax).csv'
test_filepath = 'D:/IDdataset/processedNSL/test(MinMax).csv'

dest_filepath_train = 'D:/IDdataset/processedNSL/train(135-100).csv'
dest_filepath_test = 'D:/IDdataset/processedNSL/test(135-100).csv'
pic_filepath = '../pic/AE-NSL-129-100.png'


def train_AE_129_100(train, test, is_train, is_polt):
    # 定义自编码器模型
    input_data = Input(shape=(129,))
    hidden_layer = Dense(100, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(input_data)
    output_data = Dense(129, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(hidden_layer)
    autoencoder = Model(inputs=input_data, outputs=output_data)
    encoder = Model(inputs=input_data, outputs=hidden_layer)
    # autoencoder.summary()
    if is_train == 1:
        # 训练自编码器模型
        autoencoder.compile(optimizer='adam', loss='mse')
        h = autoencoder.fit(train, train, epochs=20, batch_size=64)
        autoencoder.save('AENSL-129-100.h5')
        if is_polt == 1:
            history = h.history
            epochs = range(len(history['loss']))
            plt.plot(epochs, history['loss'])
            plt.xlabel("Epochs")
            plt.ylabel("Reconstruction Error")
            plt.rcParams["figure.dpi"] = 300
            plt.savefig(pic_filepath, dpi=300)
            plt.show()

    if is_train == 0:
        # 生成训练集
        enc_data = encoder.predict(train)
        enc_data = np.around(enc_data,5)
        pd.DataFrame(enc_data).to_csv(dest_filepath_train,header=False, index=False)
        print("shape of enc_train:" + str(enc_data.shape))
        # 生成测试集
        enc_test = encoder.predict(test)
        enc_test = np.around(enc_test, 5)
        pd.DataFrame(enc_test).to_csv(dest_filepath_test, header=False, index=False)
        # dec_data = AE.predict(data)
        # dec_data = np.around(dec_data,5)
        # pd.DataFrame(dec_data).to_csv('../dataset/KDDCUP99/Encoded/leaky-leaky.csv',header=False, index=False)
        print("shape of enc_test:" + str(enc_test.shape))


if __name__ == '__main__':
    df_train = pd.read_csv(train_filepath,header=0)
    train = df_train.values
    # print(np.where(np.isnan(train))[1])
    # print(df_train)

    df_test = pd.read_csv(test_filepath,header=0)
    test = df_test.values

    train_AE_129_100(train=train, test=test, is_train=0, is_polt=0)
