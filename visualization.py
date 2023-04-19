import pandas
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import numpy as np
import seaborn as sns

df = pd.read_csv('D:/IDdataset/processedNSL/train(onehot append char label) .csv', header=0)

pairplt = sns.pairplot(df,x_vars=['src_bytes'],y_vars=['dst_bytes'],hue='label',palette={'#F27970', '#54B345','#05B9E2','#BB9727','#696969'})

plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = (1,1)
plt.show()


