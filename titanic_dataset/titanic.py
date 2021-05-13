import pandas as pd
import numpy as np

train = pd.read_csv('titanic/train.csv')
test = pd.read_csv('titanic/test.csv')
#Note that Cabin and Age feautures are missing
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

numericalFeatures = ['Fare','Age','SibSp','Parch']
categoricalFeatures = ["Pclass", "Sex", "Embarked"]
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]

numPipeline = Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('std',StandardScaler()) 
])
from sklearn.compose import ColumnTransformer

full_pipeline = ColumnTransformer([("num",numPipeline,numericalFeatures),("cat",OneHotEncoder(),categoricalFeatures)])

yTrain = train['Survived']
Xtrain = train.drop('Survived',axis=1)
Xprep = full_pipeline.fit_transform(Xtrain)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]
classifier = RandomForestClassifier()
search = GridSearchCV(classifier,param_grid,cv=5)
score = cross_val_score(search,Xprep,yTrain,cv=10)
print("Score using forest classifier: ",score.mean())

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5]}]
search = GridSearchCV(classifier,param_grid,cv=3)
score = cross_val_score(search,Xprep,yTrain,cv=10)
print("Score using KNN classifier: ",score.mean())