import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import preprocessing
import random as random


sigma = 1.25
mu = 0
p = .95
m = 1000
test = [.01,.03,.1,.3,1,3,10,30,35]
def modelFunction(x):
    """
    :param x: Input of the function, can be matrix, scalar, or vector because it is element-wise
    :return: The output of the model function, between -.5 and .5
    """
    return 1/(1+np.exp(-x))-.5

vModel = np.vectorize(modelFunction) #makes the model function element-wise

def sigmaOutput(mu,sigma):
    """

    :param mu: The mean of the distribution
    :param sigma: The standard deviation of the distribution
    :return: Outputs a plot of the histogram for a 1D model function

    """
    gaussian = np.random.normal(mu,sigma,1000)
    position = vModel(gaussian)
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
    X_pos = np.array(vModel(X))
    Y_pos = np.array(vModel(Y))
    Locations = np.column_stack((X_pos.flatten('F'),Y_pos.flatten('F')))
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
    X_pos = np.array(vModel(X))
    Y_pos = np.array(vModel(Y))
    Locations = np.column_stack((X_pos.flatten('F'),Y_pos.flatten('F')))
    return Locations



def get_mean_power_matrix(sensing_dev_loc, integral_loc_mtx, nuc_pointing_directions, antenna_type):
    """

    :param sensing_dev_loc: A matrix with an x column and y column
    :param integral_loc_mtx: A matrix with an x row and y row
    :param nuc_pointing_directions: Don't know yet
    :param antenna_type: String omnidirectional or directional
    :return: RSSI values in a row for each sensing device
    """
    # Pt = 60  # Pt - Transmit power from mobiles for 802.11 ??? 60 dBm is 1 kW CHECK
    Pt = 20  # Pt - Transmit power from mobiles for 802.11
    fc = 2.462 * (10**9)  # fc - frequency of operation, assuming channel 11 of 802.11

    A = Pt + (20*(np.log10((3*(10**8))/float(fc)))) - (20*(np.log10(4*np.pi)))   # A - mean decay parameter
    B = -20   # B - mean decay parameter
    gain = 0  # considering equal gains on all antennas : Gain is function of the incident angle.

    ns = len(sensing_dev_loc)     # Number of sensing devices
    dist_mtx = np.zeros((ns, len(integral_loc_mtx[0])))  # returns four rows as four sensing devices
    exp_power_at_loc = np.zeros((ns,len(integral_loc_mtx[0])))  # returns four rows as four sensing devices
    if antenna_type == 'omnidirectional':

        for i in range(0, ns, 1):

            dist_mtx[i, :] = (np.sqrt((integral_loc_mtx[1,:]-sensing_dev_loc[i,1])**2 + (integral_loc_mtx[0,:]-sensing_dev_loc[i,0])**2))  # distance is a row array
            exp_power_at_loc[i, :] = A + (B * np.log10(dist_mtx[i, :])) + gain

    if antenna_type == 'directional':

        for i in range(0, ns, 1):

            x_matrix = integral_loc_mtx[0, :] - sensing_dev_loc[i, 0]
            y_matrix = integral_loc_mtx[1, :] - sensing_dev_loc[i, 1]

            dist_mtx[i, :] = np.sqrt(y_matrix**2 + x_matrix**2)  # distance is a row array
            nuc_to_loc_angle = np.arctan2(y_matrix, x_matrix) * 180 / np.pi
            nuc_to_loc_angle = np.where(nuc_to_loc_angle > 0, nuc_to_loc_angle, nuc_to_loc_angle+360)      # row vector  > 0 comparision works for float but not equality

#            exp_power_at_loc[i, :] = A + (B * np.log10(dist_mtx[i, :])) + get_Antenna_Gain(nuc_to_loc_angle, nuc_pointing_directions[i])  # gain returns a row vector

    # print "\n-------- dist loc ----------"
    # print exp_power_at_loc[i,:]

    return exp_power_at_loc

def reshapePowerMatricies(power_matrix_cor,power_matrix_uncor,m):
    size = int(m/2)
    x1 =  power_matrix_cor[0:size,:]
    x2 =  power_matrix_cor[size:2*size,:]
    y1 =  power_matrix_uncor[0:size,:]
    y2 =  power_matrix_uncor[size:2*size,:]

    top = np.concatenate((x1,x2),axis=1)
    bottom = np.concatenate((y1,y2),axis=1)
    X = np.concatenate((top,bottom))
    return X

def generateData(sensors,sigma,mu,p,m):
    size = int(m/2)
    Uncorrelated = generateUncorrelatedPositions(mu,sigma,size)
    Correlated = generateCorrelatedPositions(mu,sigma,size,p)
    power_matrix_uncor = get_mean_power_matrix(sensors,Uncorrelated.transpose(),[],'omnidirectional').transpose()
    power_matrix_cor = get_mean_power_matrix(sensors,Correlated.transpose(),[],'omnidirectional').transpose()
    X = reshapePowerMatricies(power_matrix_cor,power_matrix_uncor,m)
    y = np.concatenate((np.ones((size,1)),np.zeros((size,1))))
    return [X,y]

def getCrossValidationSet(X,y):
    cvIdx = random.sample(range(len(y)), int(.25*len(y)))
    X_cv = X[cvIdx]
    y_cv = y[cvIdx]
    X = np.delete(X,cvIdx,axis=0)
    y = np.delete(y,cvIdx,axis=0)
    return [X,y,X_cv,y_cv]

def getParamaters(X,y,X_cv,y_cv):
    min_error_ratio = 1
    C = 0
    gamma = 0
    y = np.ravel(y)
    for val in test:
     for val2 in test:
        print(val,val2)
        clf = svm.SVC(kernel='rbf',gamma=(1/(2*val2**2)),C=val)
        clf.fit(X,y)
        predict = clf.predict(X_cv)
        k = 0
        for i in range(0,len(y_cv)):
            if (y_cv[i] != predict[i]):
                k = k+1
        error_ratio = k/len(y_cv)
        if error_ratio < min_error_ratio:
            C = val
            gamma = (1/(2*val2**2))
            min_error_ratio = error_ratio
    print(min_error_ratio)
    return [C,gamma]

sensors = np.array([[0,0],[1,1],[.5,.5],[-.5,-.5],[-1,-1]])

[X,y] = generateData(sensors,sigma,mu,p,m)
[X,y,X_cv,y_cv] = getCrossValidationSet(X,y)
[C,gamma]= getParamaters(X,y,X_cv,y_cv)
print(C)
print(gamma)
clf = svm.SVC(kernel='rbf',gamma=gamma,C=C)
clf.fit(X,np.ravel(y))

