import pandas as pd
import numpy as np 
#ref; https://www.youtube.com/watch?v=kft1AJ9WVDk

#defining the sigmoid function
def sigmoid(value):
    return 1 / (1+np.exp(-value))


#getting all the inputs
df = pd.DataFrame()
df = pd.read_csv("breast_cancer.csv")
#each row in the dataframe will act as an input
column_names = ["Sample code number","Clump Thickness","Uniformity of Cell Size","Uniformity of Cell Shape","Marginal Adhesion",
"Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses","Class"]



#some basic preprocessing on the inputs
df = df.drop(columns=["Unnamed: 0"]) #dropping the initial indexing column

training_output = df["Class"]
#now lets drop the class too
df = df.drop(columns=["Class"]) #dropping the initial indexing column

#we now need a numpy array 2d
training_input = df.to_numpy()
training_output = np.array([training_output]).T #we need to transpose this to get a 699x1 array

#now we need to initialize the weights with some random values
np.random.seed(1)

#synaptic weights ; we are creating a 11x1 matrix because we have 11 inputs and 1 output



