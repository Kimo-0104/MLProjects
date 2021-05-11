import numpy as np
import MLCourse.utilities as utils
import math
import random
################################################################
############Q1: Regression###################
#########################################
class Regressor:
    """
    Generic regression interface; returns random regressor
    Random regressor randomly selects w from a Gaussian distribution
    """
    def __init__(self, parameters = {}):
        self.params = parameters
        self.weights = None

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        # Learns using the traindata
        self.weights = np.random.rand(Xtrain.shape[1])

    def predict(self, Xtest):
        # Most regressors return a dot product for the prediction
        ytest = np.dot(Xtest, self.weights)
        return ytest

    def l2err(self, prediction, y):
        """ l2 error (i.e., root-mean-squared-error) RMSE http://statweb.stanford.edu/~susan/courses/s60/split/node60.html"""
        return np.linalg.norm(np.subtract(prediction, y)) / np.sqrt(y.shape[0])


class MeanPredictor(Regressor):
    """
    Returns the average target value observed; a reasonable baseline
    """
    def __init__(self, parameters = {}):
        self.params = parameters
        self.mean = None

    def learn(self, Xtrain, ytrain):
        # Learns using the traindata
        self.mean = np.mean(ytrain)

    def predict(self, Xtest):
        return np.ones((Xtest.shape[0],))*self.mean

    def get_learned_params(self):
        return {"Mean":self.mean}


# Graduate admissions dataset: ~1.0, Adagrad: ~0.8
class LinearRegression(Regressor):
    def __init__(self, parameters):
        self.params = parameters
        self.weights = None
        self.transfer = utils.sigmoid
        self.dtransfer = utils.dsigmoid
        self.stepsize_approach = self.params['stepsize_approach']
        self.epochs = self.params['num_epochs']
        self.batch_size = self.params['batch_size']

    def learn(self, X, Y):
        self.numsamples=X.shape[0]
        self.numfeatures=X.shape[1]-1
        
        
        self.weights=np.zeros(self.numfeatures)
        if self.stepsize_approach=="heuristic":
            self.gbar=1
        elif self.stepsize_approach=="adagrad":
            self.gbar=np.zeros(self.numfeatures)

        
        self.numfeatures=X.shape[1]-1

        for i in range(self.epochs):
            for i in range(random.randint(1,3)):
                randomize = np.arange(len(X))
                np.random.shuffle(randomize)
                X = X[randomize]
                Y = Y[randomize]
            miniBatches=self.makeMiniBatches(X,Y)
            miniBatches=np.array(miniBatches,dtype=object)
            for batch in miniBatches:
                gradient=self.getGradient(batch)
                if self.stepsize_approach=="heuristic":
                    stepsize=self.getStepSize(gradient)
                    self.weights=np.subtract(self.weights,stepsize*gradient)
                elif self.stepsize_approach=="adagrad":
                    stepsize=self.getStepSize(gradient)
                    self.weights=np.subtract(self.weights,np.multiply(self.stepsize,gradient))
                    

    def getGradient(self,batch):
        sum=0
        for sample in batch:
            x=sample[0][0:self.numfeatures]
            y=sample[1]
            gradient=(np.dot(x,self.weights)-y)
            gradient=gradient*x
            sum+=gradient

        sum=sum/len(batch)
        return (sum)

    def makeMiniBatches(self, X, Y):
        miniBatches=[]
        self.numsamples=X.shape[0]
        i=0
        while(i+self.batch_size<self.numsamples):
            batch=[]
            for k in range(i,i+self.batch_size):
                batch.append([X[k],Y[k]])
            miniBatches.append(batch)
            i+=self.batch_size
        batch=[]

        for j in range(i,self.numsamples):
            batch.append([X[j],Y[j]])
        miniBatches.append(batch)
        return miniBatches

    def getStepSize(self, gradinet):
        """
        returns step size based on the approach chosen
        """
        if self.stepsize_approach=="heuristic":
            summ=0
            divisor=1

            for i in gradinet :
                summ+=abs(i)
                divisor+=1

            summ=summ/divisor
            self.gbar=self.gbar+(summ)
            self.stepsize=1/(1+self.gbar)

            return self.stepsize

        elif self.stepsize_approach=="adagrad":
            self.gbar=np.add(self.gbar,np.square(gradinet))
            self.stepsize=np.zeros(self.numfeatures)
            for i in range(self.numfeatures):
                self.stepsize[i]=1/math.sqrt(self.gbar[i])
            return self.stepsize






    def predict(self, Xtest):
        """
        Most regressors return a dot product for the prediction
        """
        #ytest=np.dot(Xtest,self.weights)
        ytest=np.empty(0)
        for sample in Xtest:
            ytest=np.append(ytest,np.dot(sample[0:self.numfeatures],self.weights))

        return ytest


# Graduate admissions dataset: ~1.0, AdaGrad: ~0.7
class PolynomialRegression(LinearRegression):
    """docstring for Po"""
    def __init__(self,parameters):
        super().__init__(parameters)

    def learn(self, X, Y):
        self.numsamples=X.shape[0]
        self.numfeatures=X.shape[1]-1
        self.Xsquared=[]

        for i in range(self.numsamples):
            x=X[i]
            m=np.square(x)
            x=np.concatenate((x[0:self.numfeatures],m))
            self.Xsquared.append(x)
        

        self.Xsquared=np.array(self.Xsquared)
        LinearRegression.learn(self,self.Xsquared,Y)  

    def predict(self, Xtest):
        self.XtestSquared=[]
        for i in range(150):
            x=Xtest[i]
            m=np.square(x)
            x=np.concatenate((x[0:Xtest.shape[1]-1],m))
            self.XtestSquared.append(x)
        self.XtestSquared=np.array(self.XtestSquared)
        return LinearRegression.predict(self,self.XtestSquared) 
        


################################################################
############Q3: Classification###################
#########################################

# Susy: ~50 error
class Classifier:
    def __init__(self, parameters = {}):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        """ Learns using the training data """
        pass

    def predict(self, Xtest, threshold=0.5):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs, threshold=threshold)
        return ytest


# Susy: ~24 error
class LogisticRegression(Classifier):
    def __init__(self, parameters):
        self.params = parameters
        self.weights = None
        self.transfer = utils.sigmoid
        self.dtransfer = utils.dsigmoid
        self.epochs = self.params['epochs']
        self.batch_size = self.params['batch_size']



    def learn(self, X, Y):
        """
        implements SGD updates
        """

        self.numsamples=X.shape[0]
        self.numfeatures=X.shape[1]
        
        
        self.weights=np.zeros(self.numfeatures)
        self.gbar=np.zeros(self.numfeatures)

        
        self.numfeatures=X.shape[1]

        for i in range(self.epochs):
            for i in range(random.randint(1,3)):
                randomize = np.arange(len(X))
                np.random.shuffle(randomize)
                X = X[randomize]
                Y = Y[randomize]
            miniBatches=self.makeMiniBatches(X,Y)
            miniBatches=np.array(miniBatches,dtype=object)
            for batch in miniBatches:
                gradient=self.getGradient(batch)
                stepsize=self.getStepSize(gradient)
                self.weights=np.subtract(self.weights,np.multiply(self.stepsize,gradient))
        pass
    def makeMiniBatches(self, X, Y):
        miniBatches=[]
        self.numsamples=X.shape[0]
        i=0
        while(i+self.batch_size<self.numsamples):
            batch=[]
            for k in range(i,i+self.batch_size):
                batch.append([X[k],Y[k]])
            miniBatches.append(batch)
            i+=self.batch_size
        batch=[]

        for j in range(i,self.numsamples):
            batch.append([X[j],Y[j]])
        miniBatches.append(batch)
        return miniBatches

    def getStepSize(self, gradinet):
        """
        returns step size based on the approach chosen
        """
        self.gbar=np.add(self.gbar,np.square(gradinet))
        self.stepsize=np.zeros(self.numfeatures)
        for i in range(self.numfeatures):
            self.stepsize[i]=1/math.sqrt(self.gbar[i])
        return self.stepsize

    def getGradient(self,batch):
        sum=0
        for sample in batch:
            x=sample[0][0:self.numfeatures]
            y=sample[1]
            gradient=(utils.sigmoid(np.dot(x,self.weights))-y)
            gradient=gradient*x
            sum+=gradient

        sum=sum/len(batch)
        return (sum)

    def predict(self, Xtest, threshold=0.5):
        Ytest=[]
        for sample in Xtest:
            sigprediction=utils.sigmoid(np.dot(sample,self.weights))
            if sigprediction<0.5:
                Ytest.append([0.])
            else:
                Ytest.append([1.])
        Ytest=np.array(Ytest)
        return Ytest
        


# Susy: ~22 error
class PolynomialLogisticRegression(LogisticRegression):
    """docstring for Po"""
    def __init__(self, parameters):
        super(PolynomialLogisticRegression, self).__init__(parameters)
    def learn(self, X, Y):
        self.numsamples=X.shape[0]
        self.numfeatures=X.shape[1]
        self.Xsquared=[]

        for i in range(self.numsamples):
            x=X[i]
            m=np.square(x)
            x=np.concatenate((x[0:self.numfeatures],m))
            self.Xsquared.append(x)
        

        self.Xsquared=np.array(self.Xsquared)
        LogisticRegression.learn(self,self.Xsquared,Y)  

    def predict(self, Xtest):
        self.XtestSquared=[]
        for i in range(self.numsamples):
            x=Xtest[i]
            m=np.square(x)
            x=np.concatenate((x[0:Xtest.shape[1]],m))
            self.XtestSquared.append(x)
        self.XtestSquared=np.array(self.XtestSquared)
        return LogisticRegression.predict(self,self.XtestSquared) 
