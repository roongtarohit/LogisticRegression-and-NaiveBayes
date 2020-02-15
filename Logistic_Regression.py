import scipy.io
import numpy
import math
import datetime

print("\n ######################### LOGISTIC REGRESSION ######################### ")

starttime = datetime.datetime.now()
print("\nSTARTED : ", starttime, "\n")

# ******** Loading the file ********************* #
MNIST_data= scipy.io.loadmat('mnist_data.mat')

# *********** TRAINING DATA SET ***************** #
training_input_X = numpy.array(MNIST_data['trX'])
training_output_Y = numpy.array(MNIST_data['trY'])

# ************** TESTING DATA SET **************** #
tsX = numpy.array(MNIST_data['tsX'])
tsY = numpy.array(MNIST_data['tsY'])

# ************ FEATURE EXTRACTION FUNCTION *******************

def feature_extraction(training_input_X):
    features = []                         # Multi-dimensional List (mean and standard_deviation)
    for index in range(0,len(training_input_X)):
        mean = sum(training_input_X[index])/len(training_input_X[index])
        standard_deviation = 0
        for feature in range(0, len(training_input_X[index])):
            standard_deviation += (training_input_X[index][feature] - mean)**2
        standard_deviation /= len(training_input_X[index])
        features.append([1,mean,math.sqrt(standard_deviation)])  # X0 is always 1
    return features


def sigmoid_function(THETA,X):
    return 1.0/(1.0+numpy.exp(-numpy.dot(X,THETA)))

def gradient_ascent(THETA, X, Y):
    
    print("DEFAULT VALUE OF THETA \n", THETA)
    
    start = datetime.datetime.now()
    print("\nGradient Ascent Started at ", start)
    change_cost=1
    alpha=0.001                                        # Learning rate
    count = 0
    while change_cost > 0.08 and count<80000:
        output_error = Y -sigmoid_function(THETA,X)
        gradient = numpy.dot(X.T, output_error)
        THETA = THETA+alpha*gradient
        change_cost = numpy.absolute(numpy.mean(output_error))
        count+=1
    
    #print(change_cost)
    end = datetime.datetime.now()
    print("Gradient Ascent Ended at ", end ," after ", count, " iterations")
    print("Gradient Ascent Duration : ", end-start)

    print("\nFINAL VALUE OF THETA \n", THETA)

    return THETA

def prediction(tsX, tsY, THETA):
    
    correct1=[0] * 2                               # To store accuracy for each digit
    total1=[0] * 2
    
    count = 0
    correct = 0
    numberOfTestingSet = len(tsY[0])
    features = feature_extraction(tsX)
    for index in range(0,numberOfTestingSet):
        output = 0
        probability = sigmoid_function(THETA,features[index])
        #print("For feature ", features[index], " the probability is ", probability)
        if probability > 0.5:
            output=1
        if output == tsY[0][index]:
            correct+=1
            correct1[int(tsY[0][index])]+=1
        
        total1[int(tsY[0][index])]+=1
        count+=1
    
    for label in range(2):                            # For loop to print accuracy for each digit
        if label == 0:
            print("\nDIGIT 7 : ")
        else:
            print("\nDIGIT 8 : ")
        print("Result - ",correct1[int(label)], " out of ", total1[int(label)])
        print("Accuracy - ",correct1[int(label)]*100/total1[int(label)])
    
    print("\nRESULT : ", correct, " out of ", count)
    print("ACCURACY : ", correct*100/count)

# ************** DATA DECLARATION AND INITIALIZATION ******************* #

X=feature_extraction(training_input_X)                     # New features
Y=training_output_Y
n=len(X[0])                                                # Number of features
m=len(Y[0])                                                # Number of training sets
THETA=numpy.ones(shape=(n,1))
X=numpy.matrix(X)                                          # m x (n+1)
Y=numpy.matrix(Y).T

# ************** CALLING THE MAIN FUNCTION ******************* #

prediction(tsX, tsY,gradient_ascent(THETA, X,Y))

endtime = datetime.datetime.now()
print("\nENDED : ", endtime)
print("\n*********** DURATION :", endtime - starttime, " *********** ")

