import matplotlib.pyplot as plt
import numpy as np
import random

# Data set extraction by deleting first and last columns
data = np.loadtxt('./data.txt', delimiter=',', usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9))
dataorig=np.loadtxt('./data.txt', delimiter=',')

# Initializing k values
k = [2]# 3, 4, 5, 6, 7, 8]
plot = np.zeros(len(k))   #array of 0's floating point value
size = data.shape[0]
print("[INFO]:Size is ",size)

def k_means():
    for i in k:
        centroid = np.zeros([i, 9])   #2 dimensional array of ix9 zero's
        print("[INFO]:Initially Centroid is",centroid)
        for j in range(i): 
            # Dividing the data from mid point into two parts
            data1 = random.randint(1, int(size / 2))
            data2 = random.randint(int(size / 2), size - 1) 
            centroid[j] = np.average(data[data1:data2], axis=0) #we take the average of all the values in a particular column
            print("[INFO]:Splitting")
        
        print("[INFO]:Finally Centroid is",centroid) #we basically have 2 centroids here #each with 9 components
            

        # Initializing distance and creating copy of centroids with zeros
        distance = np.zeros([size, i]) #699x2
        centroid_copy = np.zeros(centroid.shape)
        #np.sum sums up the value of every single element in the array
        while np.sum(centroid - centroid_copy) != 0:
            centroid_copy[:, :] = centroid[:, :]

            for j in range(i):
                distance[:, j] = np.linalg.norm((data - centroid[j]), axis=1) #normal vector returned
                print("DISTANCE:",distance)
                classification = np.argsort(distance, axis=1)
                print("CLASSIFICATION:",distance)
                classification = np.delete(classification, np.arange(1, i), 1)

                arrange = np.reshape(np.argsort(classification, axis=0), data.shape[0])

                data1 = 0
                potential = 0
                for j in range(data.shape[0] - 1):
                    if classification[arrange[j]] != classification[arrange[j + 1]]:
                        data2 = j + 1
                        centroid[classification[arrange[j]]] = np.average(data[arrange[data1:data2]], axis=0)
                        potential += np.sum(np.square(
                            np.linalg.norm(data[arrange[data1:data2]] - centroid[classification[arrange[j]]], axis=1)))
                        data1 = data2

                centroid[classification[arrange[j - 1]]] = np.average(data[arrange[data1:data.shape[0]]], axis=0)
                potential += np.sum(np.square(
                    np.linalg.norm(data[arrange[data1:data.shape[0]]] - centroid[classification[arrange[j - 1]]],
                                   axis=1)))
            plot[k.index(i)] = potential
    print(plot)


k_means()