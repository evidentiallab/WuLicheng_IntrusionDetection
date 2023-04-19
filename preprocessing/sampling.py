import pandas as pd
from collections import Counter
from imblearn.over_sampling import RandomOverSampler, SMOTE

train_filepath = 'D:/IDdataset/processedNSL/train(MinMax).csv'
train_label_filepath = 'D:/IDdataset/train_label.csv'
label_filepath = 'D:/IDdataset/processedNSL/concatenated.csv'

df_train = pd.read_csv(train_filepath,header=0)
# print(df_train.shape)
train_label = pd.read_csv(label_filepath,header=0)
train_label = train_label['label'].loc[:125972]
# print(train_label.shape)
print(Counter(train_label))

smote = SMOTE(random_state=0)
train_smoted,label_smoted = smote.fit_resample(df_train,train_label)
print(Counter(label_smoted))
train_smoted.to_csv('D:/IDdataset/processedNSL/SMOTE/NSL_train_MinMax.csv',index=False,header=True)
label_smoted.to_csv('D:/IDdataset/processedNSL/SMOTE/NSL_train_MinMax_label.csv',index=False,header=True)
