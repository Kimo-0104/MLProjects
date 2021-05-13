from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
def load_data():
    mnist = fetch_openml('mnist_784',version=1)
    return mnist
def display_image(image):
    image = image.values.reshape(28,28)
    plt.imshow(image,cmap=mpl.cm.binary,interpolation='nearest')
    plt.axis('off')
    plt.show()
def split_data(X,y):
    Xtrain, Xtest, yTrain, yTest = train_test_split(X,y,random_state=0,train_size=60000)
    return Xtrain, Xtest, yTrain, yTest

def SGD_binary_classifier(Xtrain,Xtest,yTrain,yTest):
    yTrain_5 = yTrain==5
    yTest_5 = yTest==5
    from sklearn.linear_model import SGDClassifier
    classifier = SGDClassifier(random_state=0)
    classifier.fit(Xtrain,yTrain_5)
    from sklearn.model_selection import StratifiedKFold
    from sklearn.base import clone
    '''
    The stratified k fold class performs stratified sampling to produce folds that contain a representative ratio of each class
    '''
    skfolds=StratifiedKFold(n_splits=3,random_state=0,shuffle=True)
    for trainIndex, testIndex in skfolds.split(Xtrain,yTrain_5):
        clone_clf = clone(classifier)
        X_train_folds = Xtrain.iloc[trainIndex]
        y_train_folds = yTrain_5.iloc[trainIndex]
        X_test_fold = Xtrain.iloc[testIndex]
        y_test_fold = yTrain_5.iloc[testIndex]
        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_fold)
        n_correct = sum(y_pred == y_test_fold)
        print("Accuracy of fold:",n_correct / len(y_pred))
    #The Code above is almost equivalent to:
    from sklearn.model_selection import cross_val_score
    print("Accuracy of each fold:",cross_val_score(classifier,Xtrain,yTrain_5,cv=3,scoring='accuracy'))
    '''
    We notice the classifer has 94% accuracy however this doesnt mean its good, since only about 10% of data is a 5
    so if you were to always guess not 5 you would get 90% accuary. Accuracy is not a good measure for performance on
    skewed datasets (some classes more frequent than others)

    A better way to evaluate performance of a classifier is by looking at the confusion matrix. An implementation below:
    '''
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import confusion_matrix, precision_score, recall_score,f1_score
    #This performs kfold cross validation like above, but instead of returning evaluation scores it returns the predictions
    #Made on each test fold, meaning we get clean predictions for each instances in the training set.
    y_train_pred = cross_val_predict(classifier,Xtrain,yTrain_5,cv=3)
    confusionMatrix = confusion_matrix(yTrain_5,y_train_pred)
    print("Confusion matrix:", confusionMatrix)
    '''
    Matrix obtained: [53471 1079]
                     [ 2078  3372]
    First row: Non 5 images: 53471(TN) were correctly classified as non 5 images and 1079(FP) were wrongly classified as non 5 images
    Second row: images of 5's : 2078 were classified as 5's when they arent (FP) and 3372 were truly classified as 5's (TP)
    A perfect classifier would have only true posititves and true negatives => only non zero diagonal
    '''
    print("Precision:{}\nRecall:{}\nF-1 Score:{}".format(precision_score(yTrain_5,y_train_pred),recall_score(yTrain_5,y_train_pred),f1_score(yTrain_5,y_train_pred)))
    '''
    The F1 score is the harmonic mean of precision and recall
    Precision and recall is a trade off: more of one often leads to less of the other
    '''

    '''
    One can make a classifier favor precision over recall or vice versa. The SGDClassifier makes its classification based on a decision function.
    It computes a score with this decision function and if that score if higher than a threshold it assigns the instance to a positive class
    So intuitively, one can lower this threshold to favor recall as more instances will be classified as true and vice versa.

    Example:
    '''

    a_5 = Xtrain.iloc[9]
    # the decsion_function method returns a score for each instance and then makes predictions based on the scores 
    # The SGD classifier uses 0 as a threshold
    a_5_score = classifier.decision_function([a_5])
    print("Score of the 5",a_5_score)
    #We can print the scores of all targets:
    yScores = cross_val_predict(classifier,Xtrain,yTrain,cv=3,method='decision_function')
    '''
    from sklearn.metrics import precision_recall_curve
    precisions , recalls , thresholds  = precision_recall_curve(yTrain_5,yScores)

    Suppose we want 90% precision, we can search for the lowest threshold that gives us atleast 90% precision
    (np.argmax() gives us the first index of the maximum value,which in this case is the first true value)
    print("here")
    threshold90 = thresholds[np.argmax(precisions>=90)]
    yTrain90 = (yScores>=threshold90)
    print("Precision after adjusting threshold:",(yTrain_5,yTrain90))
    '''

def random_forest_classifier(Xtrain,Xtest,yTrain,yTest):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    '''
    cross_val_predict() performs K-fold cross-validation, but instead of returning the evaluation scores, it returns the predictions 
    made on each test fold
    '''
    yTrain_5 = yTrain==5
    classifier = RandomForestClassifier(random_state=42)
    yProbs = cross_val_predict(classifier,Xtrain,yTrain_5,cv=3,method='predict_proba')
    yScores = yProbs[:, 1] 
    print(roc_auc_score(yTrain_5, yScores))


'''
All the previous functions were examples of binary classifiers.
Here is an example of multiclass classifiers
'''
def MultiClassClassifiers(Xtrain,Xtest,yTrain,yTest):
    '''
    A multiclass classifier can distinguish between more than two classes
    unlike the binary classifier.

    Some algos such as random forest classifers or NB classifiers can
    handle multiple features directly.

    However SVM classifiers and linear classifiers are strictly binary
    classifiers. However there are strategies to use to perform multiclass
    classification using multiple binary classifiers

    SVM classifiers scale poorly with the size of the training set, so a 
    one versus one approach is preferred since it is faster
    to train many classifiers on small training sets, compared to training
    few classifiers on large sets

    For most binary classification algorithms. One versus all is preferred
    '''
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.multiclass import OneVsOneClassifier
    sgd_classifier = SGDClassifier(random_state=42)
    #Scikit learn detects when you try using a binary classification algorithm
    #For multi class classification and automatically runs one versus all
    #Except for SVMS where it runs one versus one
    sgd_classifier.fit(Xtrain,yTrain)
    a_5 = Xtrain.iloc[9]
    #Here scikit learn trained 10 binary classifiers got their decision 
    #scores for the image and selected the class with the highest score
    print(sgd_classifier.predict([a_5]))
    #One can force scikit_learn to use OvO or Ova as follows
    OvO_sgd_classifier = OneVsOneClassifier(SGDClassifier(random_state=42))

    #Training a RandomForestClassifier
    rfc_classifer = RandomForestClassifier(random_state=42)
    rfc_classifer.fit(Xtrain,yTrain)
    rfc_classifer.predict([a_5])
    #Scikit learn doesnt have to run OvA or OvO since Random Forest Classifers
    #Dirrectly classify instances into multiple classes.
    #One can call predict_proba() to get the list of the probabilities
    #that the classifier assigned to each instance for each class
    print(rfc_classifer.predict_proba([a_5]))

    from sklearn.model_selection import cross_val_score
    #For evaluation we want to use cross validation, lets evaluate
    #SGDClassifiers accuracy.
    print("SGD accuracy on 3 folds: ",cross_val_score(sgd_classifier,Xtrain,yTrain,cv=3,scoring='accuracy'))

    #We can improve by scaling the inputs
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Xtrain2 = scaler.fit_transform(Xtrain.astype(np.float64))
    print("SGD accuracy on 3 folds after scaling: ",cross_val_score(sgd_classifier,Xtrain2,yTrain,cv=3,scoring='accuracy'))
def MultiLabelClassification(Xtrain,Xtest,yTrain,yTest):
    '''
    Untill now weve only considered cases in which a training example
    is assigned to only one class. IN some cases we may want our classifier
    to to output multiple classes for each instance
    '''
    from sklearn.neighbors import KNeighborsClassifier

    largeYTrain = (yTrain>=7)
    oddYTrain = (yTrain % 2 ==1)
    multiLabelYTrain = np.c_[largeYTrain,oddYTrain]

    classifier = KNeighborsClassifier()
    classifier.fit(Xtrain,multiLabelYTrain)
    a_5 = Xtrain.iloc[9]
    print(classifier.predict([a_5]))
    


if __name__=='__main__':
    mnist = load_data()
    X, y = mnist['data'], mnist['target']
    y = y.astype(np.uint8)
    Xtrain, Xtest, yTrain, yTest = split_data(X,y)
    #SGD_binary_classifier(Xtrain,Xtest,yTrain,yTest)
    #random_forest_classifier(Xtrain,Xtest,yTrain,yTest)
    MultiClassClassifiers(Xtrain,Xtest,yTrain,yTest)

