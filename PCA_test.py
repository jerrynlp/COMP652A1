__author__ = 'Administrator'
import numpy as np
from sklearn.decomposition import PCA
from numpy import linalg as LA
import matplotlib.pyplot as plt
Inputs = "/home/2015/wzhang77/652/A3/hw3pca.txt"
X = np.loadtxt(Inputs)
np.random.shuffle(X)
X1, X2, X3, X4, X5 = np.split(X[:240], 5)
Train = np.concatenate((X1, X2, X3, X4), axis=0)
Test = X5
train_construction_error = []
test_construction_error = []
variance = []
for n in range(1, Train.shape[1] + 1):
    pca = PCA(n_components=n)
    pca.fit(Train)
    Diff = Train - pca.inverse_transform(pca.transform(Train))
    construction_error = 0.0
    for i in range(Train.shape[0]):
        construction_error += LA.norm(Diff[:, i])
    train_construction_error.append(construction_error)
    Diff = Test - pca.inverse_transform(pca.transform(Test))
    construction_error = 0.0
    for i in range(Test.shape[0]):
        construction_error += LA.norm(Diff[:, i])
    test_construction_error.append(construction_error)
    variance.append(np.sum(pca.explained_variance_ratio_))
plt.ylabel('Reconstruction Error')
plt.xlabel('Number of Dimensions')
plt.plot(range(1, Train.shape[1] + 1), train_construction_error, 'r', label='train')
plt.plot(range(1, Train.shape[1] + 1), test_construction_error, 'b', label='test')
plt.legend(loc='best')
plt.show()

plt.ylabel('Reconstruction Error')
plt.xlabel('Variance')
plt.plot(variance, train_construction_error, 'r', label='train')
plt.plot(variance, test_construction_error, 'b', label='test')
plt.legend(loc='best')
plt.show()