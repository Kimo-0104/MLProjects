import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
iris = datasets.load_iris()
features = iris['data']
target = iris['target'].astype(np.float64)

svm_classifier = Pipeline([
    ('std',StandardScaler()),
    ('SVC',LinearSVC(C=1,loss='hinge'))
])

svm_classifier.fit(features,target)
#88%
print(cross_val_score(svm_classifier,features,target,cv=3,verbose=3).mean())
