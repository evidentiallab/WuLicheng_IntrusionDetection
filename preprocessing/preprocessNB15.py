import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler,StandardScaler

if __name__ == '__main__':
    train_df = pd.read_csv('D:/NB15dataset/UNSW_NB15_trainingset.csv',header=0)
    test_df = pd.read_csv('D:/NB15dataset/UNSW_NB15_testingset.csv',header=0)
    concat_df = pd.concat([train_df, test_df], ignore_index=True)
    label_df = concat_df['attack_cat']

    concat_df = concat_df.drop(['id','proto','service','state','attack_cat','label'],axis=1)

    # label_df = pd.read_csv('D:/NB15dataset/concat_label.csv',header=0)
    # label_one_hot = pd.get_dummies(label_df['attack_cat'], prefix="label")
    # label_df = label_df.drop("attack_cat", axis=1)
    # label_df = label_df.join(label_one_hot)
    # label_df.to_csv('D:/NB15dataset/concat_label_onehot.csv',header=True,index=False)

    # scaler = MinMaxScaler()
    scaler = StandardScaler()

    onehot_1 = pd.get_dummies(concat_df['is_ftp_login'], prefix='is_ftp_login')
    concat_df = concat_df.drop('is_ftp_login',axis=1)
    concat_df = concat_df.join(onehot_1)

    onehot_2 = pd.get_dummies(concat_df['is_sm_ips_ports'],prefix='is_sm_ips_ports')
    concat_df = concat_df.drop('is_sm_ips_ports', axis=1)
    concat_df = concat_df.join(onehot_2)

    standardscale_data = scaler.fit_transform(concat_df)

    np.savetxt('D:/NB15dataset/standardscale_concat.csv',standardscale_data,delimiter=',',fmt='%.5f')






