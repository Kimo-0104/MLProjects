"""
A3: Q1
"""
import numpy as np
import argparse
from argparse import RawTextHelpFormatter

import MLCourse.dataloader as dtl
import algorithms as algs


# Error functions
def l2err_squared(prediction, ytest):
    return np.square(np.linalg.norm(np.subtract(prediction, ytest)))

def get_error(predictions, ytest):
    # Can change this to other error values
    return l2err_squared(predictions, ytest)

if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description='Arguments for running.')
    parser = argparse.ArgumentParser(description='Arguments for running', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--trainsize', type=int, default=350,
                        help='Specify the train set size')
    parser.add_argument('--testsize', type=int, default=150,
                        help='Specify the test set size')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Specify the number of eposchs')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Specify the number of samples in each batch')
    parser.add_argument('--seednumber', type=int, default=1234550,
                        help='Specify the seednumber')

    args = parser.parse_args()
    trainsize = args.trainsize
    testsize = args.testsize
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    seed_number = args.seednumber

    trainset, testset = dtl.load_grad_admission()
    Xtrain, Ytrain = trainset
    Xtest, Ytest = testset
    np.random.seed(seed_number)

    algos = {
        'Mean Predictor': algs.MeanPredictor,
        'Random Predictor': algs.Regressor,
        'Linear Regression Heuristic': algs.LinearRegression,
        'Linear Regression AdaGrad': algs.LinearRegression,
        'Polynomial Regression Heuristic': algs.PolynomialRegression,
        'Polynomial Regression AdaGrad': algs.PolynomialRegression,
    }
    numalgs = len(algos)

    # Specify the name of the algorithm and an array of parameter values to try
    # if an algorithm is not include, will run with default parameters
    parameters = {
        'Mean Predictor':[],
        'Random Predictor':[],
        'Linear Regression Heuristic':{'stepsize_approach': 'heuristic' , 'num_epochs': int(num_epochs/2), 'batch_size': batch_size},
        'Linear Regression AdaGrad':{'stepsize_approach': 'adagrad' , 'num_epochs': int(num_epochs/2), 'batch_size': batch_size},
        'Polynomial Regression Heuristic': {'stepsize_approach': 'heuristic' , 'num_epochs': num_epochs, 'batch_size': batch_size},
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
        print ('Running learner = ' + learnername + ' on parameters ' + str(params))
        # Train model with best params
        learner.learn(Xtrain, Ytrain)
        # Test model with best params
        predictions = learner.predict(Xtest)
        error = get_error(predictions, Ytest)
        print ('Error for ' + learnername + ': ' + str(error))
        errors[learnername] = error

    print('\n')

    for learnername in algos:
        print('\nError for ' + learnername + ': ' + str(errors[learnername]))
