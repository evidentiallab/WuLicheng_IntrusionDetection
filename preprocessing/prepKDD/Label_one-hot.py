import pandas as pd


def label_one_hot(df):
    label_one_hot = pd.get_dummies(df["label"],prefix="label")
    df = df.drop("label", axis=1)
    df = df.join(label_one_hot)
    return df


if __name__ == '__main__':
    df = pd.read_csv('../../dataset/KDDCUP99/Label/testLabel/binary-label.csv', header=0)
    df = label_one_hot(df)
    df = df.reindex(columns=['label_normal.','label_attack.'])
    # print(df)
    label_set = df.to_csv('../dataset/KDDCUP99/Label/testLabel/binary-label-onehot.csv', header=True, index=None)


    # df2 = pd.read_csv('../dataset/KDDCUP99/Label/testLabel/multi-label.csv',header=0)
    # df2 = label_one_hot(df2)
    # df2 = df2.reindex(columns=['label_normal.', 'label_Dos.','label_Probe.', 'label_U2R.', 'label_R2L.'])
    # # print(df)
    # label_set2 = df2.to_csv('../dataset/KDDCUP99/Label/testLabel/multi-label-onehot.csv', header=True, index=None)

