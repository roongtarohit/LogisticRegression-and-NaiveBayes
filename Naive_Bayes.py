import scipy.io
import numpy
import math
import datetime
from scipy.stats import multivariate_normal

print(" ######################### NAIVE BAYES ######################### ")
starttime = datetime.datetime.now()
print("\nSTARTED : ", starttime, "\n")

# ******** Loading the file ********************* #
MNIST_data= scipy.io.loadmat('mnist_data.mat')

# *********** TRAINING DATA SET ***************** #
training_input_X = numpy.array(MNIST_data['trX'])
training_output_Y = numpy.array(MNIST_data['trY'])

# ************** TESTING DATA SET **************** #
testing_input_X = numpy.array(MNIST_data['tsX'])
testing_output_Y = numpy.array(MNIST_data['tsY'])

# ************ FEATURE EXTRACTION FUNCTION *******************

def feature_extraction(X):
    features = []                         # Multi-dimensional List (mean and standard_deviation)
    for index in range(0,len(X)):
        mean = sum(X[index])/len(X[index])
        standard_deviation = 0
        for feature in range(0, len(X[index])):
            standard_deviation += (X[index][feature] - mean)**2
        standard_deviation /= len(X[index])
        features.append([mean,math.sqrt(standard_deviation)])  # X0 is always 1
    return features

# ************** MEAN CALCULATION FUNCTION ********************

def meanOfFeatures(label, X, Y):
    sum = [0] * d                               # list of 'd' mean (number of features)
    count = 0
    for index in range(0,n):
        if Y[0][index] == label:
            for feature in range(0,len(X[index])):
                sum[feature] += X[index][feature]
            count += 1
    #print("Count : ",count)
    sum = [x/count for x in sum]
    return sum

# ************** STANDARD DEVIATION FUNCTION ********************

def stdOfFeatures(label, X, Y):
    
    sum = [0] * d                               # list of 'd' st. deviation (number of features)
    count = 0
    for index in range(0,n):
        if Y[0][index] == label:
            for feature in range(0,len(X[index])):
                sum[feature] += (X[index][feature] - mean[label][feature])**2
            count += 1
    sum = [math.sqrt(x/count) for x in sum]
    return sum

# ************** PRIOR PROBABILITY FUNCTION ********************

def prior_probability(label, Y):
    count=0
    total=0
    for output in Y[0]:
        if output == label:
            count+=1
        total+=1
    return count/total

# ************** COVARIANCE FUNCTION ********************

def covariance_matrix(label,X,Y):
    
    matrix = numpy.zeros(shape=(d,d))     #Square matrix n x n [number of training set]
    
    for index in range(0,d):             # Since independent variable, it would be a diagonal matrix
        matrix[index][index] = std[label][index]**2
    return matrix

# ************ MULTIVARIATE **************

def multivariate(tsX, X, Y, label):

    tsX=numpy.matrix(tsX)
    x_u = tsX.T-mean_transpose[label]
    
    exp_value=numpy.dot((x_u).T,covariance_inverse[label])
    exp_value=numpy.dot((exp_value),x_u)
    denominator = ((numpy.sqrt((2*math.pi)**d))*numpy.sqrt(covariance_det[label]))
    numerator = numpy.exp(-exp_value*0.5)
    return numerator/denominator
    
    """
    ALTERNATIVE WAY TO CALCULATE THE MULTIVARIATE - USING INBUILT FUNCTION
    print("By formula" , multivariate_normal.pdf(tsX,mean=mean[int(label)],cov=covariance[int(label)]))
    
    Important link : http://cs229.stanford.edu/section/gaussians.pdf
    """

def prediction(tsX, tsY):

    correct1=[0] * uniqueOutput
    total1=[0] * uniqueOutput
    
    correct = 0
    count = 0
    numberOfTestingSet = len(tsY[0])
    features = feature_extraction(tsX)
    for index in range(0,numberOfTestingSet):
        probability = -1
        #print(" FEATURES OF INPUT VECTOR  ", features[index])
        #print("************* Expected Output : ", tsY[0][index])
        output = -1
        for label in labels:
            #print("Against label : ", label)
            #print("Prior : ", prior[int(label)])
            current = multivariate(features[index],X,training_output_Y,int(label))*prior[int(label)]
            #print("Probability : ", current)
            if current > probability:
                probability = current
                output = label
        
        #print("******** Predicted output is : ", output, " with ", probability)
        if output == tsY[0][index]:
            correct+=1
            correct1[int(tsY[0][index])]+=1

        total1[int(tsY[0][index])]+=1
        count+=1
    
    
    
    for label in labels:                            # For loop to print accuracy for each digit
        if label == 0:
            print("DIGIT 7 : ")
        else:
            print("DIGIT 8 : ")
        print("Result - ",correct1[int(label)], " out of ", total1[int(label)])
        print("Accuracy - ",correct1[int(label)]*100/total1[int(label)])

    print("\nTOTAL RESULT : ", correct, " out of ", count)
    print("ACCURACY : ", correct*100/count)

# ************** DATA DECLARATION AND INITIALIZATION ******************* #

d=int(0)                                         # Number of features
n=int(0)                                         # Number of training sets
X = []                                           # training input set X
Y = []                                           # training output set X
labels = []                                      # Unique Label
covariance = []                                  # convariance matrix for every label
covariance_inverse = []                          # inverse of convariance matrix for every label
covariance_det=[]                                # determinant of convariance matrix for every label
mean_transpose=[]                                # mean transponse for every label
prior=[]                                         # prior probability for every label
mean = []                                        # mean for every label
std = []                                         # standard deviation matrix for every label
          
X=feature_extraction(training_input_X)           # New features
Y=training_output_Y

d=len(X[0])                                      # Number of features
n=len(Y[0])                                      # Number of training sets

for label in Y[0]:
    if label not in labels:
        labels.append(label)

uniqueOutput = len(labels)                       # Number of Unique Output

for index in range (d):
    mean.append(meanOfFeatures(index, X, Y))
    std.append(stdOfFeatures(index, X, Y))
    prior.append(prior_probability(index,Y))
    covariance.append(covariance_matrix(index,X,Y))
    covariance_inverse.append(numpy.linalg.inv(covariance[index]))
    covariance_det.append(numpy.linalg.det(covariance[index]))
    mean_transpose.append(numpy.matrix(mean[index]).T)

"""
for index in range (d):
    print("***************************")
    print(mean[index])
    print(std[index])
    print(prior[index])
    print(covariance[index])
    print(covariance_inverse[index])
    print(covariance_det[index])
"""

# ************** CALLING THE MAIN FUNCTION ******************* #
prediction(testing_input_X,testing_output_Y)

endtime = datetime.datetime.now()
print("\nENDED : ", endtime)
print("\n*********** DURATION :", endtime - starttime, " *********** ")


