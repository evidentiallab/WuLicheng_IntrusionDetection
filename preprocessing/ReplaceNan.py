import pandas as pd
import numpy as np
# pd.set_option('display.max_rows', None)


df = pd.read_csv('../dataset/KDDCUP99/concatenated/train-MinMax.csv', header=0)
print(df[df.isnull().values == True])

# print(df.iloc[:,[14]])
# df.fillna(0, inplace=True)
# print(df.iloc[:,[14]])
