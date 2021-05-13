import pandas as pd
import numpy as np
#Q1
def proportion_of_education():
    df = pd.read_csv('assets/NISPUF17.csv')
    ltHighSchool = df[df['EDUC1'] == 1]
    eqHighSchool = df[df['EDUC1'] == 2]
    gtHighSchool = df[df['EDUC1'] == 3]
    college = df[df['EDUC1']== 4]
    nltHighSchool = len (ltHighSchool)
    neqHighSchool = len (eqHighSchool)
    ngtHighSchool = len (gtHighSchool)
    ncollege= len(college)

    total = nltHighSchool + neqHighSchool + ngtHighSchool + ncollege

    pltHighSchool = nltHighSchool / total
    peqHighSchool = neqHighSchool / total
    pgtHighSchool = ngtHighSchool / total
    pcollege= ncollege / total

    answer = {"less than high school":pltHighSchool,
            "high school":peqHighSchool,
            "more than high school but not college":pgtHighSchool,
            "college":pcollege}
    return answer
#Q2
def average_influenza_doses():
    df = pd.read_csv('assets/NISPUF17.csv')
    #Breasfed children
    breastfed = df[df['CBF_01']==1]
    #Non brestfed children
    notBreastFed = df[df['CBF_01']==2]
    breastFedAverage = np.mean(breastfed['P_NUMFLU'])
    nonBreastFedAverage  = np.mean(notBreastFed['P_NUMFLU'])
    return (breastFedAverage,nonBreastFedAverage)

#Q3
def chickenpox_by_sex():
    df = pd.read_csv('assets/NISPUF17.csv')
    males = df[df['SEX']==1]
    females = df[df['SEX']==2]

    vaccinatedMen = males[males['P_NUMVRC']>0]
    vaccinatedWomen = females[females['P_NUMVRC']>0]

    #Vaccinated men with chicken pox
    VMWCP = len(vaccinatedMen[vaccinatedMen['HAD_CPOX']==1])
    #Vaccinated men without chickenpox
    VMWOCP = len(vaccinatedMen[vaccinatedMen['HAD_CPOX']==2])

    #Vaccinated women with chicken pox
    VWWCP = len(vaccinatedWomen[vaccinatedWomen['HAD_CPOX']==1])
    #Vaccinated women without chickenpox
    VWWOCP = len(vaccinatedWomen[vaccinatedWomen['HAD_CPOX']==2])

    menAnswer = VMWCP/VMWOCP
    womenAnswer = VWWCP/VWWOCP

    return {'male':menAnswer, 'female':womenAnswer}



#Q4
def corr_chickenpox():
    import scipy.stats as stats
    import numpy as np
    import pandas as pd
    
    df = pd.read_csv('assets/NISPUF17.csv')

    df = df[df['P_NUMVRC']>=0]
    df = df[df['HAD_CPOX']<3]
    
    #print(df['P_NUMVRC'].unique())
    # here is some stub code to actually run the correlation
    corr, pval=stats.pearsonr(df["HAD_CPOX"],df["P_NUMVRC"])
    #print(pval)
    # just return the correlation
    return corr  
    
print(corr_chickenpox())