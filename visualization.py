import pandas
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import numpy as np
import seaborn as sns
from scipy.spatial.distance import euclidean
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from sklearn.metrics import calinski_harabasz_score

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
confusion_mtx_1 = np.array([[0.993, 0.001, 0.002, 0,    0.004],
 [0.001, 0.999, 0,    0,    0   ],
 [0.005, 0.001, 0.994, 0,    0  ],
 [0.232, 0,    0,    0.747, 0.021],
 [0.071, 0,    0.001, 0.001, 0.927]])

confusion_mtx_2 = np.array([[0.993, 0.001, 0.002, 0,    0.005],
 [0.001, 0.998, 0.001, 0,    0],
 [0.009, 0,    0.991, 0,    0   ],
 [0.232, 0,    0,    0.695, 0.074],
 [0.063, 0,    0.001, 0.002, 0.934]])

confusion_mtx_3 = np.array([[0.992, 0.001, 0.002, 0,    0.005],
 [0.001, 0.999, 0,    0,    0   ],
 [0.004, 0,    0.995, 0,    0   ],
 [0.2,   0.011, 0,    0.705, 0.084],
 [0.066, 0.001, 0.001, 0.001, 0.931]])

confusion_mtx_4 = np.array([[0.993, 0.001, 0.003, 0,    0.004],
 [0.001, 0.998, 0,    0,    0   ],
 [0.006, 0.001, 0.992, 0,    0   ],
 [0.347, 0,    0,    0.568, 0.084],
 [0.102, 0,    0.001, 0,    0.897]])

confusion_mtx_test = np.array([[0.793, 0.152, 0.018, 0.035],
 [0.136, 0.806, 0.019,    0.038,],
 [0.027, 0.037, 0.767, 0.168],
 [0.017, 0.018,    0.167,    0.802]])

# normal =confusion_mtx[0, :]
# dos =   confusion_mtx[1, :]
# probe = confusion_mtx[2, :]
# u2r =   confusion_mtx[3, :]
# r2l =   confusion_mtx[4, :]

# print("Distance of Normal-U2R = " + str(euclidean(normal, u2r)))
# print("Distance of R2L-Normal = " + str(euclidean(r2l, normal)))
# print("Distance of R2L-U2R = " + str(euclidean(r2l, u2r)))
# print("Distance of R2L-centroid = " + str(euclidean(normal, centroid)))

# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'
#
# plt.figure(figsize=(11,10))
#
# dn = dendrogram(sch.linkage(confusion_mtx_1, method='ward'),color_threshold=0,
#                 labels=['Normal','Dos','Probe','U2R','R2L'])
# print(sch.linkage(confusion_mtx_1, method='ward'))
# plt.rcParams['font.sans-serif'] = ['Times New Roman']
# plt.ylabel('Distance',fontsize=28)
# plt.xlabel('Label',fontsize=28)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.plot(linewidth=2)
# plt.show()
# # print(confusion_mtx)
# for n in [4,3,2]:
#     agg_clustering = AgglomerativeClustering(n_clusters=n)
#     agg_label = agg_clustering.fit_predict(confusion_mtx_1)
#     CHI_score = calinski_harabasz_score(confusion_mtx_1, labels=agg_label)
#     print(f"Calinski-Harabasz Index: {CHI_score:.2f}")

"""================================================================================"""

# confusion_mtx_nusw = np.array([[0.077,0,    0.089,0.646,0.039,0,    0.149,0,    0,    0   ],
#  [0,    0.094,0.113,0.725,0.042,0.001,0.007,0.009,0.007,0.001],
#  [0,    0.001,0.199,0.757,0.012,0.002,0.013,0.005,0.011,0.   ],
#  [0,    0.001,0.05, 0.889,0.013,0.003,0.021,0.02, 0.004,0.   ],
#  [0,    0,    0.01, 0.107,0.591,0.001,0.28, 0.006,0.004,0.   ],
#  [0,    0,    0.002,0.012,0.001,0.983,0.001,0,    0.001,0.   ],
#  [0.001,0,    0,    0.006,0.047,0,    0.942,0.002,0.001,0.   ],
#  [0,    0.001,0.03, 0.171,0.004,0,    0.006,0.78, 0.008,0.   ],
#  [0,    0.001,0.004,0.099,0.078,0.008,0.1,  0.058,0.652,0.   ],
#  [0,    0,    0.024,0.559,0.063,0.016,0.008,0.008,0.016,0.307]])
#
# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'
#
# plt.figure(figsize=(18,10))
#
# dn = dendrogram(sch.linkage(confusion_mtx_nusw, method='ward'),color_threshold=0,
#                 labels=['Analysis','Backdoor','Dos','Exploits','Fuzzers','Generic','Normal','Recon','Shellcode','Worms'])
# print(sch.linkage(confusion_mtx_nusw, method='ward'))
# plt.rcParams['font.sans-serif'] = ['Times New Roman']
# plt.ylabel('Distance',fontsize=28)
# plt.xlabel('Label',fontsize=28)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.plot(linewidth=2)
# plt.show()
# # print(confusion_mtx)
# for n in [9,8,7,6,5,4,3,2]:
#     agg_clustering = AgglomerativeClustering(n_clusters=n)
#     agg_label = agg_clustering.fit_predict(confusion_mtx_nusw)
#     CHI_score = calinski_harabasz_score(confusion_mtx_nusw, labels=agg_label)
#     print(f"Calinski-Harabasz Index: {CHI_score:.2f}")
print("**************************集值标签数量=" + str(2) + "**************************")