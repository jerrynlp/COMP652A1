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


def experiment(X, y):
    ##################################################################################
    # Split the data randomly
    ##################################################################################
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


    ##################################################################################
    # Run linear regression on the data using L2 regularization
    ##################################################################################
    lr = LinearRegression()
    _lambda = [0]
    plot_lambda = []
    train_rmse = []
    test_rmse = []
    for para in _lambda:
        plot_lambda.append(log(para, 10))
        lr.fit(X_train, y_train, para)
        y_predict = lr.predict(X_train)
        train_rmse.append(rmse(y_train, y_predict))
        y_predict = lr.predict(X_test)
        test_rmse.append(rmse(y_test, y_predict))
    '''
    plt.xlabel('log(lambda)')
    plt.ylabel('RMSE')
    plt.plot(plot_lambda, train_rmse, 'r')
    plt.plot(plot_lambda, test_rmse, 'b')
    plt.show()
    '''
    return train_rmse, test_rmse


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
_lambda = [0.1, 1, 10, 100, 1000]
plot_lambda = []
train_rmse = []
test_rmse = []
weigh_vector_L2 = []
weigh_vector_1 = []
weigh_vector_2 = []
weigh_vector_3 = []
weigh_vector_4 = []
for para in _lambda:
    plot_lambda.append(log(para, 10))
    lr.fit(X_train, y_train, para)
    y_predict = lr.predict(X_train)
    train_rmse.append(rmse(y_train, y_predict))
    y_predict = lr.predict(X_test)
    test_rmse.append(rmse(y_test, y_predict))
    # Cal L2-norm of weight vector
    weigh_vector_L2.append(L2(lr.weight_vector()))
    weigh_vector_1.append(lr.weight_vector()[0])
    weigh_vector_2.append(lr.weight_vector()[1])
    weigh_vector_3.append(lr.weight_vector()[2])
    weigh_vector_4.append(lr.weight_vector()[3])

plt.xlabel('log(lambda)')
plt.ylabel('RMSE')
plt.plot(plot_lambda, train_rmse, 'r')
plt.plot(plot_lambda, test_rmse, 'b')
plt.show()
'''
plt.plot(plot_lambda, weigh_vector_L2, 'r')
plt.show()

plt.plot(plot_lambda, weigh_vector_1, 'r')
plt.plot(plot_lambda, weigh_vector_2, 'b')
plt.plot(plot_lambda, weigh_vector_3, 'y')
plt.plot(plot_lambda, weigh_vector_4, 'b')
plt.show()
'''
##################################################################################
# Re-format the data
##################################################################################
_mu = [-9, -7, -5, -3, -1, 1, 3, 5, 7, 9]
_sigma = [0.1, 0.5, 1, 5, 10]
for sigma in _sigma:
    new_X = constant
    for mu in _mu:
        for i in range(0, 3):
            gaussian_x = [gaussian(x, mu, sigma) for x in X[:, i]]
            temp = np.asarray(gaussian_x, dtype='float32').reshape(new_X.shape[0], 1)
            new_X = np.append(new_X, temp, axis=1)
    experiment(new_X, y)

