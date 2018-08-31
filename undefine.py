#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:23:25 2018

@author: user
"""
import numpy as np
import json
import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
from scipy.spatial.distance import cdist
from scipy.spatial import distance
from sklearn import metrics,cross_validation
from sklearn.metrics import mean_squared_error
from math import sqrt

dataset=pd.read_csv("RFM.csv")
X=dataset.iloc[:,[1,2,3]].values
data=pd.read_csv("RFM3.csv")
r=dataset.iloc[:,0].values
y=data.iloc[:,[1,2,3]].values
cluster_num = 3


kmeans = KMeans(n_clusters=cluster_num)

kmeans.fit(X)
kmeans.predict(X)
#X_train, X_test,y_train,y_test=  cross_validation.train_test_split(X,y,test_size=0.20,random_state=70)
#score = metrics.accuracy_score(kmeans.predict(y_test),kmeans.predict(X_test))
#mi=metrics.mutual_info_score(X,y)
#print(mi)
#print('Accuracy:{0:f}'.format(score))
#rms=sqrt(mean_squared_error(y_test),y_pred)
#print('Root Mean Square:{0:f}'.format(rms))
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
#print(labels)

colors = ["g.","r.","c."]
d = Counter(labels)
dict={'label_0':{},'label_1':{},'label_2':{}}
dict=json.dumps(dict)

fig = figure()
ax = fig.gca(projection='3d')
for i in range(len(X)):
#    print("coordinate:",X[i], "label:", labels[i])
    
    if labels[i]==0:
        
         ax.scatter(X[i][0], X[i][1], X[i][2],c='r')
    elif labels[i]==1:
         ax.scatter(X[i][0], X[i][1], X[i][2],c='y')
    
    else:
         ax.scatter(X[i][0], X[i][1], X[i][2],c='g')
        
print("Clustering of Customers based on RFM score","\n")
#for cluster_number in range(cluster_num):
 
#  print("Cluster {} contains {} samples ".format(cluster_number , d[cluster_number]))

ax.scatter(centroids[:, 0],centroids[:, 1], centroids[:, 2], marker = "x", s=150, linewidths = 5, zorder = 100,c='b')

centroids=[centroids[:, 0],centroids[:, 1], centroids[:, 2]]


plt.show()


def ClusterIndicesNumpy(clustNum, labels_array): #numpy 
    return np.where(labels_array == clustNum)[0]

def closest_centroid(rfm, centroids):
    min_distance = float('inf')
    belongs_to_cluster = None
    for j in range(len(centroids)):
        centroid = centroids[j]
        dist = distance.euclidean(rfm, centroid)
        if dist < min_distance:
            min_distance = dist
            belongs_to_cluster = j

    return belongs_to_cluster





    
    
rfm=[]
rfm.append(int(input("Enter R value")))
rfm.append(int(input("Enter F value")))
rfm.append(int(input ("Enter M value")))
#print("RFM",rfm)
c=closest_centroid(rfm, centroids)
p={}
p['cluster_number']=c
        
p['Member_Id']=r[ClusterIndicesNumpy(c,kmeans.labels_)]
print(p)
#def ClusterIndicesComp(clustNum, labels_array): #list comprehension
#    return np.array([i for i, x in enumerate(labels_array) if x == clustNum])
#print(ClusterIndicesComp(2, kmeans.labels_))




