import pandas
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import numpy as np
import seaborn as sns
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from math import dist

# df = pd.read_csv('D:/IDdataset/processedNSL/train(onehot append char label) .csv', header=0)
#
# pairplt = sns.pairplot(df,x_vars=['src_bytes'],y_vars=['dst_bytes'],hue='label',palette={'#F27970', '#54B345','#05B9E2','#BB9727','#696969'})
#
# plt.rcParams['figure.dpi'] = 300
# plt.rcParams['figure.figsize'] = (1,1)
# plt.show()
# for font in font_manager.fontManager.ttflist:
#     # 查看字体名以及对应的字体文件名
#     print(font.name, '-', font.fname)
confusion_mtx = np.array([
                            [0.969, 0.006, 0.023, 0, 0.001],
                            [0.037, 0.941, 0.021, 0, 0],
                            [0.103, 0,0.895, 0.002, 0],
                            [0.541, 0,0,0.324, 0.135],
                            [0.754, 0,    0.003, 0,0.244]])

test = np.array([
                            [0.793, 0.152, 0.018, 0.035],
                            [0.136, 0.806, 0.019, 0.038],
                            [0.027, 0.037, 0.767, 0.168],
                            [0.017, 0.018, 0.167, 0.802],])

normal = [0.969, 0.006, 0.023, 0, 0.001]
dos = [0.037, 0.941, 0.021, 0, 0]
probe = [0.103, 0,0.895, 0.002, 0]
u2r = [0.541, 0,0,0.324, 0.135]
r2l = [0.754, 0,    0.003, 0,0.244]
print('normal-r2l = '+str(dist(normal,r2l)))
print('normal-u2r = '+str(dist(normal,u2r)))
print('u2r-r2l = '+str(dist(r2l,u2r)))

# confusion_mtx_PR = np.array([[0.97,  0.008, 0.021, 0.001, 0],
#                             [0.027, 0.962, 0.011, 0,0],
#                             [0.112, 0.002, 0.886, 0, 0],
#                             [0.459, 0,0, 0.459, 0.081],
#                             [0.791, 0, 0.002, 0,    0.206]])
#
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


plt.figure(figsize=(8,7))

dn = dendrogram(sch.linkage(test,method='ward'),color_threshold=0,
                labels=['Normal','Dos','Probe','U2R' ])
print(sch.linkage(test,method='ward'))
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.ylabel('Euclidean distances',fontsize=17)
# plt.xlabel(xlabel=['Normal','Dos','Probe','U2R','R2L'],fontproperties='Times New Roman')
plt.show()
print(confusion_mtx)