import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.optimize import minimize
from scipy.optimize import fmin
from sklearn.feature_extraction import DictVectorizer
import sklearn.cluster
import distance
import os
import time

df = pd.read_csv('company_list.csv')

list = df['0'].tolist()

dict = {}
words = np.asarray(list)
lev_similarity = -1*np.array([[distance.levenshtein(w1,w2) for w1 in words] for w2 in words])

affprop = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=0.5)
affprop.fit(lev_similarity)
for cluster_id in np.unique(affprop.labels_):
    exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
    cluster = np.unique(words[np.nonzero(affprop.labels_==cluster_id)])
    cluster_str = ", ".join(cluster)
    dict[exemplar] = cluster_str
