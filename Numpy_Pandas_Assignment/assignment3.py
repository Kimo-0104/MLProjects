import pandas as pd
import numpy as np
from string import digits
import string
import re
import warnings
warnings.filterwarnings('ignore')
def Q1():
    #Reading in file, replacing all ... with NAN
    df = pd.read_excel('assets/Energy Indicators.xls',na_values='...')
    #Deleting the useless columns
    del df['Unnamed: 0']
    del df['Unnamed: 1']
    #Deleting useless rows
    rowstodrop=np.concatenate((np.arange(0,17,1),np.arange(244,282,1)))
    df.drop(df.index[rowstodrop],inplace=True)
    #Renaming columns to meaningfull names
    newColumnNames={'Unnamed: 2':'Country','Unnamed: 3':'Energy Supply','Unnamed: 4':'Energy Supply per Capita','Unnamed: 5':'% Renewable'}
    df.rename(columns=newColumnNames,inplace=True)
    #Convering petajoules into gigajoules
    df['Energy Supply']=df['Energy Supply'].apply(lambda x: x*1000000)
    #Making country the new index
    df.set_index('Country',inplace=True)
    #To view index we use loc and iloc
    #print(df.loc['Republic of Korea'])

    #Renaming countries
    newCountryNames={"Republic of Korea": "South Korea",
            "United States of America20": "United States",
            "United Kingdom of Great Britain and Northern Ireland19": "United Kingdom",
            "China, Hong Kong Special Administrative Region3": "Hong Kong"}
    df.rename(newCountryNames,inplace=True)

    indexList=df.index.tolist()
    countriesToRename={}
    for oldCountry in indexList:
        newCountry=re.sub("[\(\[].*?[\)\]]", "", oldCountry).translate(str.maketrans('', '', digits)).rstrip()
        countriesToRename[oldCountry]=newCountry
    df.rename(countriesToRename,inplace=True)
    
    columnsNeeded=np.concatenate((['Country Name'],np.arange(2006,2016,1)))
    GDPInfo = pd.read_csv('assets/world_bank.csv',skiprows=4)[columnsNeeded].set_index('Country Name')
    countriesToRename={"Republic of Korea": "South Korea",
                    "United States of America": "United States",
                    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
                    "China, Hong Kong Special Administrative Region": "Hong Kong"}
    GDPInfo.rename(countriesToRename,inplace=True)
    newCountries={'Hong Kong SAR, China':'Hong Kong',
                   'Iran, Islamic Rep.':'Iran',
                    'Korea, Rep.':'South Korea'}
    GDPInfo.rename(newCountries,inplace=True)
    ScimEn = pd.read_excel('assets/scimagojr-3.xlsx')[:15]
    df1 = pd.merge(ScimEn, df, how = 'inner', left_on = 'Country', right_on='Country')
    #print(df1.head(15))
    final = pd.merge(df1, GDPInfo, how = 'inner', left_on = 'Country', right_on='Country Name').set_index('Country')
    return final

#What are the top 15 countries for average GDP over the last 10 years?
def Q3(df):
    columnsNeeded=['2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']
    meandf=df[columnsNeeded]
    meandf['mean']=np.mean(meandf,axis=1)
    mean=meandf['mean'].sort_values(ascending=False)
    return mean

#By how much had the GDP changed over the 10 year span for the country with the 6th largest average GDP?
def Q4(df):
    columnsNeeded=['2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']
    GDPs=df[columnsNeeded]
    GDPs['mean']=np.mean(GDPs,axis=1)
    GDPs=GDPs.sort_values('mean',ascending=False)
    sixth=GDPs.iloc[5]
    result = sixth['2015']-sixth['2006']
    return result

#What is the mean energy supply per capita?
def Q5(df):
    energySupplies=df['Energy Supply per Capita']
    mean = np.mean(energySupplies)
    print(mean)

#What country has the maximum % Renewable and what is the percentage?
def Q6(df):
    percentRenew=df['% Renewable']
    result = np.max(percentRenew)
    country = df[df['% Renewable']==result].index[0]
    return (country,result)

#Create a new column that is the ratio of Self-Citations to Total Citations. 
#What is the maximum value for this new column, and what country has the highest ratio?
def Q7(df):
    columnsNeeded=['Self-citations','Citations']
    newDf=df[columnsNeeded]
    newDf['ratio']=newDf['Self-citations']/newDf['Citations']
    newDf-newDf.sort_values('ratio',ascending=False)
    maxVal=np.max(newDf['ratio'])
    country=newDf[newDf['ratio']==maxVal].index[0]
    return (country,maxVal)
#Create a column that estimates the population using Energy Supply and Energy Supply per capita. 
#What is the third most populous country according to this estimate?
def Q8(df):
    columnsNeeded=['Energy Supply','Energy Supply per Capita']
    newDf=df[columnsNeeded]
    newDf['PopEstimate']=newDf['Energy Supply']/newDf['Energy Supply per Capita']
    newDf=newDf.sort_values('PopEstimate',ascending=False)
    thirdPop=newDf.iloc[2]['PopEstimate']
    country=newDf[newDf['PopEstimate']==thirdPop].index[0]
    return country
#What is the correlation between the number of citable documents per capita 
#and the energy supply per capita? Use the .corr() method, (Pearson's correlation). 
def Q9(df):
    df['PopEstimate']=df['Energy Supply']/df['Energy Supply per Capita']
    df['Citable docs per Capita']=df['Citable documents']/df['PopEstimate']
    docs=df['Citable docs per Capita'].astype('float')
    energy=df['Energy Supply per Capita'].astype('float')
    corr = docs.corr(energy)
    return corr
#Create a new column with a 1 if the country's % Renewable value is at or above the median for all countries in the top 15,
# and a 0 if the country's % Renewable value is below the median.
def Q10(df):
    newDf=df['% Renewable'].sort_values(ascending=False)
    median=np.median(newDf)
    df['HighRenew'] = [1 if x >= median else 0 for x in df['% Renewable']]
    return df['HighRenew']
#Use the following dictionary to group the Countries by Continent, then create a DataFrame that displays 
#the sample size (the number of countries in each continent bin), 
#and the sum, mean, and std deviation for the estimated population of each country.
def Q11(df):
    ContinentDict  = {'China':'Asia', 
                  'United States':'North America', 
                  'Japan':'Asia', 
                  'United Kingdom':'Europe', 
                  'Russian Federation':'Europe', 
                  'Canada':'North America', 
                  'Germany':'Europe', 
                  'India':'Asia',
                  'France':'Europe', 
                  'South Korea':'Asia', 
                  'Italy':'Europe', 
                  'Spain':'Europe', 
                  'Iran':'Asia',
                  'Australia':'Australia', 
                  'Brazil':'South America'}
    df.reset_index(inplace=True)
    df['Continent']=[ContinentDict[x] for x in df['Country']]
    df['PopEstimate']=(df['Energy Supply']/df['Energy Supply per Capita']).astype('float')
    df = df.set_index("Continent").groupby(level=0)['PopEstimate'].agg(['size','sum','mean','std'])
    return df
def result():
    s = 'ACAABAACAAABACDBADDDFSDDDFFSSSASDAFAAACBAAAFASD'
    result = []
    # compete the pattern below
    pattern = "[\w]AAA"
    for item in re.finditer(pattern, s):
      # identify the group number below.
      result.append(item.group().strip('A'))
      
    return result
if __name__ == '__main__':
    df=Q1()
    #Q11(df)
    #print(Q11(df))
    import re
    print(result())
