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


x=scaled_df.iloc[0:400,1:10] #.values.transpose()
y=df.iloc[0:400,10:] #.values.transpose()
xval=scaled_df.iloc[401:550,1:10] #.values.transpose()
yval=df.iloc[401:550,10:] #.values.transpose()
xtest=scaled_df.iloc[551:683,1:10] #.values.transpose()
ytest=df.iloc[551:683,10:] #.values.transpose()



# # load the dataset
# dataset = loadtxt('breast_cancer_copy.csv', delimiter=',')
# # split into input (X) and output (y) variables
# X = dataset[:,0:9]
# y = dataset[:,9]


# define the keras model
model = Sequential()

#2 hidden layers with the first input of 8
#one last hidden layer
model.add(Dense(12, input_dim=9, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='relu'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(x, y, epochs=95, batch_size=10)


# evaluate the keras model
_, accuracy = model.evaluate(xval, yval)
print('Accuracy: %.2f' % (accuracy*100))


# ytest = model.predict_classes(xtest)
# print(ytest)

