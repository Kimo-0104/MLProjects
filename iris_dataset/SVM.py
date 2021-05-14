import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
iris = datasets.load_iris()
features = iris['data']
target = iris['target'].astype(np.float64)

svm_classifier = Pipeline([
    ('std',StandardScaler()),
    ('SVC',LinearSVC(C=1,loss='hinge'))
])

svm_classifier.fit(features,target)
#88%
print("Score using linear SVM: ",cross_val_score(svm_classifier,features,target,cv=3).mean())

poly_svm_classifer = Pipeline([
    ('poly',PolynomialFeatures(degree=3)),
    ('std',StandardScaler()),
    ('SVC',LinearSVC(C=10,loss='hinge'))
])

poly_svm_classifer.fit(features,target)
#96%
print("Score using polynomial SVM: ",cross_val_score(poly_svm_classifer,features,target,cv=3,verbose=3).mean())

from sklearn.svm import SVC
svm_classifier2 = Pipeline([
    ('std',StandardScaler()),
    ('SVC',SVC(C=5,kernel='poly',degree=3,coef0=1))
])

svm_classifier2.fit(features,target)
print("Score using polynomial SVM: ",cross_val_score(poly_svm_classifer,features,target,cv=3,verbose=3).mean())