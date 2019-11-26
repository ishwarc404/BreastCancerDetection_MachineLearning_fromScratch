import random
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Euclidian Distance between two d-dimensional points
def eucldist(p0,p1):
    dist = 0.0
    for i in range(0,len(p0)):
        dist += (p0[i] - p1[i])**2
    return math.sqrt(dist)


    
#K-Means Algorithm
def kmeans(k,datapoints):

    # d - Dimensionality of Datapoints
    d = len(datapoints[0]) 
    
    #Limit our iterations
    Max_Iterations = 1000
    i = 0
    
    cluster = [0] * len(datapoints)
    prev_cluster = [-1] * len(datapoints)
    
    #Randomly Choose Centers for the Clusters
    cluster_centers = []
    for i in range(0,k):
        new_cluster = []
        #for i in range(0,d):
        #    new_cluster += [random.randint(0,10)]
        cluster_centers += [random.choice(datapoints)]
        
        
        #Sometimes The Random points are chosen poorly and so there ends up being empty clusters
        #In this particular implementation we want to force K exact clusters.
        #To take this feature off, simply take away "force_recalculation" from the while conditional.
        force_recalculation = False
    
    while (cluster != prev_cluster) or (i > Max_Iterations) or (force_recalculation) :
        
        prev_cluster = list(cluster)
        force_recalculation = False
        i += 1
    
        #Update Point's Cluster Alligiance
        for p in range(0,len(datapoints)):
            min_dist = float("inf")
            
            #Check min_distance against all centers
            for c in range(0,len(cluster_centers)):
                
                dist = eucldist(datapoints[p],cluster_centers[c])
                
                if (dist < min_dist):
                    min_dist = dist  
                    cluster[p] = c   # Reassign Point to new Cluster
        
        
        #Update Cluster's Position
        for k in range(0,len(cluster_centers)):
            new_center = [0] * d
            members = 0
            for p in range(0,len(datapoints)):
                if (cluster[p] == k): #If this point belongs to the cluster
                    for j in range(0,d):
                        new_center[j] += datapoints[p][j]
                    members += 1
            
            for j in range(0,d):
                if members != 0:
                    new_center[j] = new_center[j] / float(members) 
                
                #This means that our initial random assignment was poorly chosen
                #Change it to a new datapoint to actually force k clusters
                else: 
                    new_center = random.choice(datapoints)
                    force_recalculation = True
                    print("Forced Recalculation...")
                    
            
            cluster_centers[k] = new_center
    
        
    print ("======== Results ========")
    # print ("Clusters", cluster_centers)
    # print ("Iterations",i)
    # print ("Assignments", cluster)
    return (cluster_centers,cluster)
    
    

    
    
#TESTING THE PROGRAM#
if __name__ == "__main__":
    #2D - Datapoints List of n d-dimensional vectors. (For this example I already set up 2D Tuples)
    #Feel free to change to whatever size tuples you want...
    datapoints = [(0,0,1),(0,0,0.9),(1,1,0),(1,1,1)]#,(1,0),(1,1),(5,6),(7,7),(9,10),(11,13),(12,12),(12,13),(13,13)]

    k = 2 #K - Number of Clusters
      
    print("[INFO]:DataPath Read")
    df = pd.read_csv("breast_cancer_mode_replaced.csv")
    df = df.drop("Sample code number", axis=1)
    target = pd.DataFrame()
    
    target["Class"] = df["Class"]
    df = df.drop("Class", axis=1)
    df = df.drop(df.columns[0],axis=1)
    user_dataframe = df.astype(int).values.tolist()  #converting the entire dataframe to float 
    target = target.astype(int).values
    print("[INFO]:DataFrame Returned")
    new_target = [i[0] for i in target]
    target = new_target
    distances = []

    for k in range(2,10):
        print("[INFO]:KMEANS DONE FOR K=",k)
        cluster_centers, cluster = kmeans(k,user_dataframe)
        length = len(cluster)
        # print(len(cluster_centers[0]))
        total = 0
        for i in range(length):
            a = np.asarray(user_dataframe[i])
            b = np.asarray(cluster_centers[cluster[i]])
            dist = np.linalg.norm(a-b)
            total = total + dist
        
        distances.append(total)
    
    minimum  = min(distances)
    # print("DISTANCES:",distances)
    # print("MINIMUM IS:",minimum)
    # print("HENCE OPTIMAL K VALUE IS",distances.index(minimum))
    ks = [i for i in range(2,10)]
    plt.plot(ks,distances)
    plt.show()


