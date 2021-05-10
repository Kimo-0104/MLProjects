import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.linear_model import LinearRegression
def load_data():
    return pd.read_csv('housing/housing.csv')
class DisplayData:
    def __init__(self,df):
        self.df = df
    def plot_histogram(self):
        self.df.hist(bins=50,figsize=(20,15))
        plt.show()
    def head(self):
        print(self.df.head())
    def describe(self):
        print(self.df.describe())
    def info(self):
        print(self.df.info())  
    def lat_lon_scatterplot(self,density=False):
        #The alpha paramater makes it easier to visualize the places where there is a high density
        # of data points
        if density:
            self.df.plot(kind='scatter',x='longitude',y='latitude',alpha=0.1)
        else:
            self.df.plot(kind='scatter',x='longitude',y='latitude')
        plt.show()   
    def lat_lon_houseprice_scatterplot(self):
        '''
        The radius of each circle represents the districts population (option s)
        The color represents the price (option c)
        We use a predefined color map (jet) which ranges from blue(low) to red(high)
        '''
        self.df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
            s=self.df["population"]/100, label="population", figsize=(10,7),
            c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
            )
        plt.legend()
        plt.show()
        '''
        This plot shows us that house prices is very much related to the location and the population density.
        Also shows that the ocean proximity attribute may be useful as well.
        '''
    def correlationMatrix(self):
        '''
        Correlation ranges from -1<=x<=1. A value of 1 indicates there is a strong +ve correlation.
        Example: Suppose median house value and median income are positevly correlated. this implies that
        the median house value goes up when the median income goes up

        We can compute the standard correlation coeffecient between every pair of attributes using the 
        corr() method
        '''
        corr_matrix = self.df.corr()
    
    def scatter_mattrix(self):
        '''
        Note: diagonals plots are the histograms of the corresponding feature
        '''
        features = ['median_income','housing_median_age','total_rooms','median_house_value']
        scatter_matrix(self.df[features],figsize=(12,8))
        plt.show()
def split_data_randomly(df):
    '''
    Random Sampling
    '''
    features=df.loc[:,df.columns!='median_house_value']
    target = df['median_house_value']
    #To generate the same samples every time ensure that the random_state seed is the same everytime
    Xtrain, Xtest, yTrain, yTest = train_test_split(features,target,random_state=0)
    return Xtrain,yTrain, Xtest,yTest

def split_data_stratified(df):
    '''
    The split_data_randomly function is random sampling. Random sampling is fine when it comes to very large datasets.
    However if a dataset is large we may encounter sampling bias. One way of addresseing sampling bias is using 
    stratified sampling: the population is divided into homegenous groups called strata and the right number of instances
    is sampled from each stratum to guarantee the test set is representative of the overall population

    In our case if we knew that median income is a very important feature in predicting median housing prices,we
    would want to ensure that the test set is representative of the various categories of prices.
    '''

    #If we were to look at the histogram for median income, most median incomes are clustered around 1.5 and 6
    #(15000-60000). We should not have many strata and each stratum should be large enough

    #Here we create an income category with 5 categories 1: 0-1.5, 2: 1.5-3 ......
    df['income_cat']=pd.cut(df['median_income'],bins=[0,1.5,3,4.5,6.,np.inf],labels=[1,2,3,4,5])
    
    split = StratifiedShuffleSplit(n_splits=1,test_size=0.2, random_state=0)
    #The split method in split generates indicies to split the data into training and test set.
    for train_idx, test_idx in split.split(df,df['income_cat']):
        strat_train_set = housing.loc[train_idx]
        strat_test_set = housing.loc[test_idx]
    '''
    #Now we see that this test set follows the same distribution as our population
    strat_test_set['income_cat'].hist()
    plt.show()
    '''
    #Now we drop the income_cat attribute so our data is back to its original state
    del strat_test_set['income_cat']
    del strat_train_set['income_cat']
    '''
    For returning features and targets seperately
    XTrain = strat_train_set.loc[:,strat_train_set.columns!='median_house_value']
    yTrain = strat_train_set['median_house_value']
    XTest = strat_test_set.loc[:,strat_test_set.columns!='median_house_value']
    yTest = strat_test_set['median_house_value']
    return XTrain,yTrain,XTest,yTest
    '''
    return strat_train_set,strat_test_set

def encode_categorical_column(series):
    from sklearn.preprocessing import OrdinalEncoder
    encode = OrdinalEncoder()
    encodedSeries = encode.fit_transform(series)
    encodedSeries = pd.DataFrame(encodedSeries,index=series.index, columns=series.columns)
    encodedSeries = encodedSeries['ocean_proximity']
    '''
    Problem with Ordinal encoding: ML algorithm will assume that 2 nearby values are more similar than 2 distant values
    which isnt the case for all of cases. is fine for cases such as "bad,average,good,excellent"
    Fix: one hot encoding
    '''
    from sklearn.preprocessing import OneHotEncoder
    oneHotEncoder = OneHotEncoder()
    oneHotEncodedSeries = oneHotEncoder.fit_transform(series)
    return oneHotEncodedSeries

def clean_data(df):
    from sklearn.impute import SimpleImputer
    #imputer class will replace all missing values with the median of that column
    imputer = SimpleImputer(strategy='median')
    #We remove the ocean_proximity column as its non-numerical so we cant compute its median
    #Make it a dataframe so encoding it works, needs 2d shape
    ocean=df[['ocean_proximity']]
    numericDf=df.drop('ocean_proximity',axis=1)
    #Using the imputer transform method to replace all the missing values with the median of that column
    #Note: the imputer transform method returns a numpy array hence we need to convert it back to a dataframe
    imputer.fit(numericDf)
    X=imputer.transform(numericDf)
    cleanedDf = pd.DataFrame(X,columns=numericDf.columns,index=numericDf.index)
    #cleanedDf['ocean_proximity']=encode_categorical_column(ocean)
    return cleanedDf

def split_into_features_labels(train,test):
    XTrain = train.loc[:,train.columns!='median_house_value']
    yTrain = train['median_house_value']
    XTest = test.loc[:,test.columns!='median_house_value']
    yTest = test['median_house_value']
    return XTrain,yTrain,XTest,yTest

if __name__ == '__main__':
    housing = load_data()
    dataInfo = DisplayData(housing)
    train,test = split_data_stratified(housing)
    train = clean_data(train)
    Xtrain,yTrain,Xtest,yTest = split_into_features_labels(train,test)
    regressor = LinearRegression()
    regressor.fit(Xtrain,yTrain)