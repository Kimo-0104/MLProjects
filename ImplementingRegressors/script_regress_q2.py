"""
A3: Q2
"""
import numpy as np
import argparse
import MLCourse.dataloader as dtl
from MLCourse.utilities import pdfTStudent , cdfTStudent
import algorithms as algs
import MLCourse.plotfcns as plot
import math

# Error functions
def l2err_squared(prediction, ytest):
    return np.square(np.linalg.norm(np.subtract(prediction, ytest)))

def get_error(predictions, ytest):
    # Can change this to other error values
    return l2err_squared(predictions, ytest)

# hint: refer to utilities for the Tstudent functions
"""HypothesisTesting using paired t-test"""
def tDistPValue(errorLearner, errorBaseLine):
    """
    computes pvalue using paired t-test
    return p_value
    """
    #polynomial minus linear
    difference=[]
    for i in range(errorLearner.shape[0]):
        difference.append(errorLearner[i]-errorBaseLine[i])
    ddiff=np.mean(difference)
    stddif=np.std(difference)
    sediff=stddif/(math.sqrt(errorLearner.shape[0]))
    tvalue=ddiff/sediff

    t = tvalue
    m = errorLearner.shape[0]    # dimension of the test set
    return pValueTDistPositiveTail(t=t, dof=m-1)

def checkforPrerequisites(errorLearner, errorBaseLine, learnerName, baseLearnerName):
    meanBaseline=np.mean(errorBaseLine)
    meanLearner=np.mean(errorLearner)
    varBaseline=np.var(errorBaseLine)
    varLearner=np.var(errorLearner)
    stdBaseLine=np.std(errorBaseLine)
    stdLearner=np.std(errorLearner)
    print("The mean of the baseline(Linear) is: ",meanBaseline,"\nThe mean of the learner(Polynomial) is: ",meanLearner)
    print("The var of the baseline(linear) is: ",varBaseline,"\nThe var of the learner(polynomial) is: ",varLearner)
    print("The std of the baseline(linear) is: ",stdBaseLine,"\nThe std of the learner(polynomial) is: ",stdLearner)
    #plot.plotTwoHistograms(errorBaseLine,errorLearner)
    pass

def pValueTDistPositiveTail(t, dof):
    """
    return p_value using the t distribution
    """
    return 1 - cdfTStudent(t, dof)

if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description='Arguments for running.')
    parser = argparse.ArgumentParser(description='Arguments for running', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--trainsize', type=int, default=250,
                        help='Specify the train set size')
    parser.add_argument('--testsize', type=int, default=150,
                        help='Specify the test set size')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Specify the number of eposchs')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Specify the number of samples in each batch')
    parser.add_argument('--pvalueThreshold', type=float, default=0.05,
                        help='Specify the p value threshold')
    parser.add_argument('--seednumber', type=int, default=123455,
                        help='Specify the seednumber')
    # parser.add_argument("--usePolynomial", help="Polynomial Regression is used")


    args = parser.parse_args()
    trainsize = args.trainsize
    testsize = args.testsize
    pvalueThreshold = args.pvalueThreshold
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    seed_number = args.seednumber

    trainset, testset = dtl.load_grad_admission()
    Xtrain, Ytrain = trainset
    Xtest, Ytest = testset
    predictionsDict = {'Ground Truth': Ytest}
    np.random.seed(seed_number)

    algos = {
        'Linear Regression AdaGrad': algs.LinearRegression,
        'Polynomial Regression AdaGrad': algs.PolynomialRegression,
    }
    baseLearnerName  = 'Linear Regression AdaGrad'
    numalgs = len(algos)
    # Specify the name of the algorithm and an array of parameter values to try
    # if an algorithm is not include, will run with default parameters
    parameters = {
        'Linear Regression AdaGrad':{'stepsize_approach': 'adagrad' , 'num_epochs': num_epochs, 'batch_size': batch_size},
        'Polynomial Regression AdaGrad': {'stepsize_approach': 'adagrad' , 'num_epochs': num_epochs, 'batch_size': batch_size}
    }

    # initialize the errors for each parameter setting to 0
    errors = {}
    for learnername in algos:
        errors[learnername] = 0
        
    for learnername, Learner in algos.items():
        params = parameters.get(learnername, parameters[learnername])
        # run cross validation only if paramaters are more than 1
        learner = Learner(params)
        # print ('Running learner = ' + learnername + ' on parameters ' + str(params))
        # Train model with best params
        learner.learn(Xtrain, Ytrain)
        # Test model with best params
        predictions = learner.predict(Xtest)
        predictionsDict[learnername] = predictions
        error = get_error(predictions, Ytest)
        # print ('Error for ' + learnername + ': ' + str(error))
        errors[learnername] = error

    learnerName = 'Polynomial Regression AdaGrad'
    errorBaseLine = (predictionsDict['Ground Truth'] - predictionsDict[baseLearnerName].squeeze())**2
    errorLearner = (predictionsDict['Ground Truth'] - predictionsDict[learnerName].squeeze())**2
    # Prints the mean and std values and plots the histograms to compare the errors
    print('\n')
    checkforPrerequisites(errorLearner, errorBaseLine, learnerName, baseLearnerName)
    pval = tDistPValue(errorLearner, errorBaseLine)

    if pval < pvalueThreshold:
        result = "rejected"
    else:
        result = "not rejected"
    print('\nThe p-value for {}: {}.\nThe null Hypothesis is {} with threshold p-value = {} !\n'.format(learnername, pval, result, pvalueThreshold))

    for learnername in algos:
        print('Error for ' + learnername + ': ' + str(errors[learnername]))
    print('\n')
