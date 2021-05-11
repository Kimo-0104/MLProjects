import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
def plot_histogram(yvalues):
    plt.hist(yvalues, bins=50, facecolor='green')
    plt.xlabel('Regression variable')
    plt.ylabel('Probability')
    #plt.axis([40, 160, 0, 0.03])
    plt.grid(True)

    plt.show()

def plotTwoHistograms(d1, d2, nbins = 40, legends = ["BaseLine predictor", "Proposed Predictor"]):
    '''Plots double histograms'''
    nstd = 5
    bins1 = np.linspace(d1.mean() - nstd*d1.std(), d1.mean() + nstd*d1.std(), nbins)
    bins2 = np.linspace(d2.mean() - nstd*d2.std(), d2.mean() + nstd*d2.std(), nbins)
    plt.ylabel('Counts')
    plt.title('Error histograms')
    plt.hist(d1, bins1, alpha=0.7, label= legends[0])
    plt.hist(d2, bins2, alpha=0.7, label= legends[1])
    plt.legend(loc='upper right')
    plt.show()
