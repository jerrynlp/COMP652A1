import numpy as np
from sklearn.cross_validation import train_test_split
from numpy.linalg import inv
from numpy import linalg as LA

class linear_regression:
    def __init__(self):
        self.W = np.array

    def mse(self, X, y):
        return LA.norm(np.dot(X, self.W) - y)

    def fit(self, X, y, _lambda):
        self.W = np.dot(inv(np.dot(np.transpose(X), X) + np.dot(_lambda, np.eye(X.shape[1]))), (np.dot(np.transpose(X), y)))
        print self.mse(X, y)

TrainingInputs = "/home/2015/wzhang77/Documents/COMP652/hw1x.txt"
TrainingOutputs = "/home/2015/wzhang77/Documents/COMP652/hw1y.txt"

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

lr = linear_regression()
lr.fit(X_train, y_train, 0.0)
lr.fit(X_train, y_train, 0.1)
lr.fit(X_train, y_train, 1)
lr.fit(X_train, y_train, 10)
lr.fit(X_train, y_train, 100)
lr.fit(X_train, y_train, 1000)