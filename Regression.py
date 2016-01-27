import numpy as np
from sklearn.cross_validation import train_test_split
from numpy.linalg import inv
from numpy import linalg as LA
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        self.W = np.array

    def mse(self, X, y):
        return LA.norm(np.dot(X, self.W) - y)

    def fit(self, X, y, _lambda):
        self.W = np.dot(inv(np.dot(np.transpose(X), X) + np.dot(_lambda, np.eye(X.shape[1]))), (np.dot(np.transpose(X), y)))

    def predict(self, X):
        return np.dot(X, self.W)


def mse(y_true, y_predict):
    return LA.norm(y_true - y_predict)


TrainingInputs = "/home/2015/wzhang77/Documents/COMP652/hw1x.txt"
TrainingOutputs = "/home/2015/wzhang77/Documents/COMP652/hw1y.txt"

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
        train_output_no_id = float(line.strip())
        train_outputs.append(train_output_no_id)

# Convert python lists to numpy array
X = np.asarray(train_inputs, dtype='float32')
# Add constant input
constant = np.ones((X.shape[0], 1), dtype='float32')
X = np.append(X, constant, axis=1)
y = np.asarray(train_outputs, dtype='float32')


##################################################################################
# Split the data randomly
##################################################################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


##################################################################################
# Run linear regression on the data using L2 regularization
##################################################################################
lr = LinearRegression()
_lambda = [0, 0.1, 1, 10, 100, 1000]
for para in _lambda:
    lr.fit(X_train, y_train, para)
    y_predict = lr.predict(X_train)
    print "train error " + str(mse(y_train, y_predict))
    y_predict = lr.predict(X_test)
    print "test error " + str(mse(y_test, y_predict))