import numpy as np
import pandas as pd
# sigmoid function to normalize inputs
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid derivatives to adjust synaptic weights
def sigmoid_derivative(x):
    return x * (1 - x)

# # input dataset
# training_inputs = np.array([[0,0,1],
#                             [1,1,1],
#                             [1,0,1],
#                             [0,1,1]])

# # output dataset
# training_outputs = np.array([[0,1,1,0]]).T


#########
#getting all the inputs
df = pd.DataFrame()
df = pd.read_csv("breast_cancer.csv")
#each row in the dataframe will act as an input
column_names = ["Sample code number","Clump Thickness","Uniformity of Cell Size","Uniformity of Cell Shape","Marginal Adhesion",
"Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses","Class"]

#some basic preprocessing on the inputs
df = df.drop(columns=["Unnamed: 0"]) #dropping the initial indexing column
df = df.drop(columns=["Sample code number"]) #dropping the initial indexing column

df.iloc[:,9].replace(2, 0,inplace=True)
df.iloc[:,9].replace(4, 1,inplace=True)
training_outputs = df["Class"]


#now lets drop the class too
df = df.drop(columns=["Class"]) #dropping the initial indexing column

#we now need a numpy array 2d
training_inputs = df.to_numpy()

training_outputs = np.array([training_outputs]).T #we need to transpose this to get a 699x1 array


#########

# seed random numbers to make calculation
np.random.seed(1)

# initialize weights randomly with mean 0 to create weight matrix, synaptic weights
synaptic_weights = 2 * np.random.random((9,1)) - 1

print('Random starting synaptic weights: ')
print(synaptic_weights)

# Iterate 10,000 times
for iteration in range(50000):

    # Define input layer
    input_layer = training_inputs
    # Normalize the product of the input layer with the synaptic weights
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    # how much did we miss?
    error = training_outputs - outputs

    # multiply how much we missed by the
    # slope of the sigmoid at the values in outputs
    adjustments = error * sigmoid_derivative(outputs)

    # update weights
    synaptic_weights += np.dot(input_layer.T, adjustments)

print('Synaptic weights after training: ')
print(synaptic_weights)

print("Output After Training:")
print(outputs)