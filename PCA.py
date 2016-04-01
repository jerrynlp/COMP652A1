__author__ = 'Administrator'
import numpy as np
from numpy import linalg as LA
Inputs = "C:\\Users\\Administrator\\Documents\\652\\A3\\hw3pca.txt"
X = np.loadtxt(Inputs)
np.random.shuffle(X)
X1, X2, X3, X4, X5 = np.split(X[:240], 5)
