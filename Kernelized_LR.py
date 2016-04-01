__author__ = 'Weiwei Zhang'
import numpy as np
from sklearn.cross_validation import train_test_split
from numpy import linalg as LA
import matplotlib.pyplot as plt
from math import log, sqrt, pi, exp

###########################################################################################
# kernel function
###########################################################################################
def kernel(xi, xj, d):
    return np.power(np.dot(xi, np.transpose(xj)) + 1, d)

###########################################################################################
# gradient descent
###########################################################################################
def GD(a, K, alpha, y):
    new_a = np.zeros(a.shape[0])
    while LA.norm(new_a - a) > 0.01:
        a = new_a
        for i in range(a.shape[0]):
            decent = 0.0
            if i > 0:
                for j in range(K.shape[0]):
                    try:
                        decent += alpha * (y[j] - 1/(1+exp(-np.dot(a[1:], K[i-1]) - a[0]))) * K[j][i-1]
                    except:
                        decent += alpha * y[j] * K[j][i-1]
                new_a[i] = a[i] + decent
            elif i == 0:
                for j in range(K.shape[0]):
                    try:
                        decent += (y[j] - 1/(1+exp(-np.dot(a[1:], K[i-1]) - a[0])))
                    except:
                        decent += alpha * y[j]
                new_a[i] = a[i] + alpha * decent
    return new_a

###########################################################################################
# prediction function
###########################################################################################
def predict(a, X_train, X_test, d):
    y = np.zeros(X_test.shape[0])
    for i in range(X_test.shape[0]):
        try:
            score = 1/(1+exp(-np.dot(a[1:], kernel(X_train, X_test[i], d)) - a[0]))
        except:
            score = 0
        if score >= 0.5:
            y[i] = 1
    return y
###########################################################################################
# Accuracy function
###########################################################################################
def accuracy(y_true, y_predict):
    equal = 0.0
    for i in range(y_true.shape[0]):
        if y_predict[i] == y_true[i]:
            equal += 1
    return equal / y_true.shape[0]

def k_fold_generator(X, y, k_fold, seed = 10):
    subset_size = len(X) / k_fold
    for k in range(k_fold):
        X_train = X[:k * subset_size] + X[(k + 1) * subset_size:]
        X_valid = X[k * subset_size:][:subset_size]
        y_train = y[:k * subset_size] + y[(k + 1) * subset_size:]
        y_valid = y[k * subset_size:][:subset_size]
        yield X_train, y_train, X_valid, y_valid

def average(numbers):
    total = sum(numbers)
    total = float(total)
    total /= len(numbers)
    return total

TrainingInputs = "C:\\Users\\Administrator\\Documents\\652\\A2\\hw2x.dat"
TrainingOutputs = "C:\\Users\\Administrator\\Documents\\652\\A2\\hw2y.dat"
##################################################################################
# Load the data into memory
##################################################################################
# Load all training inputs
train_inputs = []
with open(TrainingInputs, 'rb') as train_file:
    for line in train_file:
        terms = line.strip().split('  ')
        if not len(terms) == 3:
            continue
        train_input_no_id = []
        for term in terms:
            train_input_no_id.append(float(term))
        train_inputs.append(train_input_no_id)

# Load all training outputs
train_outputs = []
with open(TrainingOutputs, 'rb') as train_file:
    for line in train_file:
        train_output_no_id = int(float(line.strip()))
        train_outputs.append(train_output_no_id)

k_fold = 3
for d in [1,2,3]:
    print "d = " + str(d)
    train_acc = []
    test_acc = []
    for X_train_list, y_train_list, X_test_list, y_test_list in k_fold_generator(train_inputs, train_outputs, k_fold):
        # Convert python lists to numpy array
        X_train = np.asarray(X_train_list, dtype='float32')
        X_test = np.asarray(X_test_list, dtype='float32')
        y_train = np.asarray(y_train_list, dtype='int')
        y_test = np.asarray(y_test_list, dtype='int')

        alpha = 0.00001
        a = np.random.normal(0, 0.1, X_train.shape[0] + 1)
        K = kernel(X_train, X_train, d)
        y_predict = predict(a, X_train, X_train, d)
        y_predict = predict(a, X_train, X_test, d)
        a = GD(a, K, alpha, y_train)
        y_predict = predict(a, X_train, X_train, d)
        train_acc.append(accuracy(y_train, y_predict))
        y_predict = predict(a, X_train, X_test, d)
        test_acc.append(accuracy(y_test, y_predict))
    train_acc.append(average(train_acc))
    test_acc.append(average(test_acc))
    print train_acc
    print test_acc
'''
plt.xlabel('Iterator')
plt.ylabel('ACC')
plot1 = plt.plot(range(0, len(train_rmse)), train_rmse, 'r', label='train')
plot2 = plt.plot(range(0, len(train_rmse)), test_rmse, 'b', label='test')
plt.legend(loc='best')
plt.show()
'''
