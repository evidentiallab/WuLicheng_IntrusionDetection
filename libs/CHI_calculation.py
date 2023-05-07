from sklearn.utils import check_X_y
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans

dataframe = pd.DataFrame(data=np.random.randint(0, 60, size=(100, 10)))

score = metrics.calinski_harabasz_score(dataframe, labels)
print(score)



