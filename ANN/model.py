import tensorflow as tf
import pandas as pd
import numpy as np

df = pd.read_csv("breast_cancer.csv")


df = df.drop(columns=["Unnamed: 0"]) #dropping the initial indexing column
training_output = pd.DataFrame()
training_output["Class"] = df["Class"]
#now lets drop the class too
df = df.drop(columns=["Class"]) #dropping the initial indexing column


x_train = df.head(500).to_numpy()
y_train = training_output.head(500).to_numpy()

x_test = df.tail(100).to_numpy()
y_test = training_output.tail(100).to_numpy()




# #normalising the data
x_train  = tf.keras.utils.normalize(x_train,axis=1)
x_test  = tf.keras.utils.normalize(x_test,axis=1)


#sequential is a feed forward model
model = tf.keras.models.Sequential()

#number  parameter is no of neurons

#input layer
model.add(tf.keras.layers.Flatten())

#hidden layer
#model.add(tf.keras.layers.Dense(5,activation=tf.nn.relu))
#model.add(tf.keras.layers.Dense(5,activation=tf.nn.relu))

#output layer
#2 classifications
model.add(tf.keras.layers.Dense(2,activation=tf.nn.softmax))

#training parameters
#adam is like SGD
model.compile(optimizer='adam',loss='sparse_categorial_crossentropy',metrics=['accuracy'])

#training
model.fit(x_train,y_train,epochs=3)


