import numpy as np
from sklearn.cross_validation import train_test_split

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

print X
print y