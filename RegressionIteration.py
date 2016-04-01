__author__ = 'Administrator'
import numpy as np
from sklearn.cross_validation import train_test_split
from numpy.linalg import inv
from numpy import linalg as LA
import matplotlib.pyplot as plt
from math import log, sqrt, pi, exp
from sklearn.metrics import mean_squared_error

class LinearRegression:
    def __init__(self):
        self.W = np.array

    def fit(self, X, y, _lambda):
        self.W = np.dot(inv(np.dot(np.transpose(X), X) + np.dot(_lambda, np.eye(X.shape[1]))), (np.dot(np.transpose(X), y)))

    def predict(self, X):
        return np.dot(X, self.W)

    def weight_vector(self):
        return self.W


def rmse(y_true, y_predict):
    return sqrt(mean_squared_error(y_true, y_predict))


def L2(w):
    return LA.norm(w)


def gaussian(x, mu, sigma):
    y = exp(-1.0 * (x - mu) ** 2 / (2 * sigma ** 2))
    return y

def experiment(X, y, _lambda):
    ##################################################################################
    # Split the data randomly
    ##################################################################################
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    ##################################################################################
    # Run linear regression on the data using L2 regularization
    ##################################################################################
    lr = LinearRegression()
    lr.fit(X_train, y_train, _lambda)
    y_predict = lr.predict(X_train)
    train_rmse = rmse(y_train, y_predict)
    y_predict = lr.predict(X_test)
    test_rmse = rmse(y_test, y_predict)
    return train_rmse, test_rmse

def experiment_weight_vector(X, y, _lambda):
    ##################################################################################
    # Split the data randomly
    ##################################################################################
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    ##################################################################################
    # Run linear regression on the data using L2 regularization
    ##################################################################################
    lr = LinearRegression()
    lr.fit(X_train, y_train, _lambda)
    return lr.weight_vector()

TrainingInputs = "C:\\Users\\Administrator\\Documents\\652\\hw1x.txt"
TrainingOutputs = "C:\\Users\\Administrator\\Documents\\652\\hw1y.txt"

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

train_rmse = []
test_rmse = []
alpha = 0.1
W = np.random.rand(X_train.shape[1])
FI = np.dot(y_train.reshape(y_train.shape[0], 1), np.transpose(W.reshape(W.shape[0], 1))) / np.dot(W, np.transpose(W))
new_W = W - alpha * (np.dot(np.dot(np.transpose(FI), FI) + np.dot(0.1, np.eye(X_train.shape[1])), W) - np.dot(np.transpose(FI), y_train))
train_rmse.append(rmse(y_train, np.dot(X_train, new_W)))
test_rmse.append(rmse(y_test, np.dot(X_test, new_W)))
while L2(new_W - W) > 0.01:
    W = new_W
    FI = np.dot(y_train.reshape(y_train.shape[0], 1), np.transpose(W.reshape(W.shape[0], 1))) / np.dot(W, np.transpose(W))
    new_W = W - alpha * (np.dot(np.dot(np.transpose(FI), FI) + np.dot(0.1, np.eye(X_train.shape[1])), W) - np.dot(np.transpose(FI), y_train))
    train_rmse.append(rmse(y_train, np.dot(X_train, new_W)))
    test_rmse.append(rmse(y_test, np.dot(X_test, new_W)))

plt.xlabel('Iterator')
plt.ylabel('RMSE')
plot1 = plt.plot(range(0, len(train_rmse)), train_rmse, 'r', label='train')
plot2 = plt.plot(range(0, len(train_rmse)), test_rmse, 'b', label='test')
plt.legend(loc='best')
plt.show()