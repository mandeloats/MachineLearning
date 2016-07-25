import numpy as np
import matplotlib.pyplot as plt

sigma = 1.65
mu = 0
p = .0

def modelFunction(x):
    """
    :param x: Input of the function, can be matrix, scalar, or vector because it is element-wise
    :return: The output of the model function, between -.5 and .5
    """
    return 1/(1+np.exp(-x))-.5

vfunc = np.vectorize(modelFunction) #makes the model function element-wise

def sigmaOutput(mu,sigma):
    """

    :param mu: The mean of the distribution
    :param sigma: The standard deviation of the distribution
    :return: Outputs a plot of the histogram for a 1D model function

    """
    gaussian = np.random.normal(mu,sigma,1000)
    position = vfunc(gaussian)
    count, bins, ignored = plt.hist(position,30,normed=True)
    plt.show()

def generateUncorrelatedPositions(mu,sigma,size):
    """

    :param mu: The mean of the 4D gaussian (same for all distributions)
    :param sigma: The standard deviation of the 4D gaussian (same for all distributions)
    :param size:  The number of position points to be output
    :return:  Position matrix with X column and Y column according to an uncorrelated 4D gaussian
    """
    cov = np.identity(2) * (sigma**2)
    mean = [mu,mu]
    X = np.random.multivariate_normal(mean,cov,size)
    Y = np.random.multivariate_normal(mean,cov,size)
    X_pos = np.array(vfunc(X))
    Y_pos = np.array(vfunc(Y))
    Locations = np.column_stack((X_pos.flatten(),Y_pos.flatten()))
    return Locations

def generateCorrelatedPositions(mu,sigma,size,p):
    """
    :param mu: The mean of the 4D gaussian (same for all distributions)
    :param sigma: The standard deviation of the 4D gaussian (same for all distributions)
    :param size:  The number of position points to be output
    :param p:  The pairwise correlation between the 2 users, inserted into the Gaussian Covariance matrix
    :return:  Position matrix with X column and Y column according to an correlated 4D gaussian
    """
    cov = np.array([[1,p],[p,1]])*(sigma**2)
    mean = [mu,mu]
    X = np.random.multivariate_normal(mean,cov,size)
    Y = np.random.multivariate_normal(mean,cov,size)
    X_pos = np.array(vfunc(X))
    Y_pos = np.array(vfunc(Y))
    Locations = np.column_stack((X_pos.flatten(),Y_pos.flatten()))
    return Locations



#sigmaOutput(mu,sigma)
Uncorrelated = generateUncorrelatedPositions(mu,sigma,1000)
Correlated = generateCorrelatedPositions(mu,sigma,1000,p)
plt.scatter(Uncorrelated[:,0],Uncorrelated[:,1],c='r')
plt.scatter(Correlated[:,0],Correlated[:,1],c='b')
plt.show()
