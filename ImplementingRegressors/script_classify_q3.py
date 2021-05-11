"""
A3: Q3
"""
import numpy as np
import argparse

import MLCourse.dataloader as dtl
import algorithms as algs


def getaccuracy(ytest, predictions):
    correct = 0
    # count number of correct predictions
    correct = np.sum(ytest == predictions)
    # return percent correct
    return (correct / float(len(ytest))) * 100

def get_error(ytest, predictions):
    return (100 - getaccuracy(ytest, predictions))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Arguments for running.')
    parser.add_argument('--trainsize', type=int, default=5000,
                        help='Specify the train set size')
    parser.add_argument('--testsize', type=int, default=5000,
                        help='Specify the test set size')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Specify the number of eposchs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Specify the number of samples in each batch')
    parser.add_argument('--seednumber', type=int, default=1234550,
                        help='Specify the seednumber')

    args = parser.parse_args()
    trainsize = args.trainsize
    testsize = args.testsize
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    seed_number = args.seednumber

    classalgs = {
        'Random': algs.Classifier,
        'Logistic Regression': algs.LogisticRegression,
        'Polynomial Logistic Regression': algs.PolynomialLogisticRegression
    }
    numalgs = len(classalgs)

    # Specify the name of the algorithm and an array of parameter values to try
    # if an algorithm is not included, will run with default parameters
    parameters = {
        'Random':[],
        'Logistic Regression': { 'epochs': num_epochs, 'batch_size': batch_size},
        'Polynomial Logistic Regression': {'epochs': num_epochs, 'batch_size': batch_size},
    }

    # initialize the errors for each parameter setting to 0
    errors = {}
    for learnername in classalgs:
        errors[learnername] = 0

    trainset, testset = dtl.load_susy(trainsize, testsize)

    Xtrain = trainset[0]
    Ytrain = trainset[1]
    # cast the Y vector as a matrix
    Ytrain = np.reshape(Ytrain, [len(Ytrain), 1])

    Xtest = testset[0]
    Ytest = testset[1]
    # cast the Y vector as a matrix
    Ytest = np.reshape(Ytest, [len(Ytest), 1])

    np.random.seed(seed_number)

    for learnername, Learner in classalgs.items():
        params = parameters.get(learnername, parameters[learnername])
        # run cross validation only if paramaters are more than 1
        learner = Learner(params)
        print ('Running learner = ' + learnername + ' on parameters ' + str(params))
        # Train model with best params
        learner.learn(Xtrain, Ytrain)
        # Test model with best params
        predictions = learner.predict(Xtest)
        error = get_error(predictions, Ytest)
        print ('Error for ' + learnername + ': ' + str(error))
        errors[learnername] = error
