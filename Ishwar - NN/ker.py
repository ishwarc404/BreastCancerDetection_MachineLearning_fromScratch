# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools



df = pd.read_csv("breast_cancer.csv")
df = df.drop(columns=["Unnamed: 0"]) #dropping the initial indexing column
df.iloc[:,10].replace(2, 0,inplace=True)
df.iloc[:,10].replace(4, 1,inplace=True)

df = df.astype(float)


names = df.columns[0:10]
scaler = MinMaxScaler() 
scaled_df = scaler.fit_transform(df.iloc[:,0:10]) 
scaled_df = pd.DataFrame(scaled_df, columns=names)


x=scaled_df.iloc[0:500,1:10] #.values.transpose()
y=df.iloc[0:500,10:] #.values.transpose()
xval=scaled_df.iloc[501:600,1:10] #.values.transpose()
yval=df.iloc[501:600,10:] #.values.transpose()
xtest=scaled_df.iloc[600:683,1:10] #.values.transpose()
ytest=df.iloc[600:683,10:] #.values.transpose()


print(x,type(x))
# # load the dataset
# dataset = loadtxt('breast_cancer_copy.csv', delimiter=',')
# # split into input (X) and output (y) variables
# X = dataset[:,0:9]
# y = dataset[:,9]


# define the keras model
model = Sequential()

#2 hidden layers with the first input of 9
#one last hidden layer
model.add(Dense(12, input_dim=9, activation='relu')) #input_dim is the number of cols in the inputs
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='relu'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#fit the keras model on the dataset
model.fit(x, y, epochs=100, batch_size=10)


# evaluate the keras model
_, accuracy = model.evaluate(xval, yval)
print('Accuracy: %.2f' % (accuracy*100))


# evaluate the keras model
_, accuracy = model.evaluate(xtest, ytest)
print('Accuracy: %.2f' % (accuracy*100))


