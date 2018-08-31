#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 14:38:15 2018

@author: user
"""

import medium
import undefine
import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
from functools import reduce

products = pd.read_csv( 'products1.csv', index_col='ProductId')

#orders = pd.read_csv( 'orders1.csv', usecols=['Profit','MemberId','ProfitCat'],index_col='MemberId')
orders = pd.read_csv( 'orders1.csv', usecols=['Profit','MemberId','ProfitCat'])

#item_prior = pd.read_csv( 'order_products_prior1.csv',index_col='MemberId')
item_prior = pd.read_csv( 'order_products_prior1.csv')

customers_data = pd.read_csv("age1.csv")
#print(customers_data)
user_product=pd.merge(pd.merge(orders ,item_prior,on='MemberId'),customers_data,on='MemberId')

#user_product = orders.join(item_prior, how='inner',lsuffix='_left', rsuffix='_right').reset_index().groupby(['MemberId','ProductId']).count()

user_product = user_product.reset_index().rename(columns={'Profit':'prior_order_count'})

from scipy.sparse import csr_matrix
user_product_sparse = csr_matrix((user_product['prior_order_count'], (user_product['MemberId'], user_product['ProductId'])), shape=(user_product['MemberId'].max()+1, user_product['ProductId'].max()+1), dtype=np.uint16)
user_product_sparse = csr_matrix((user_product ['prior_order_count'], (user_product['MemberId'],user_product['ProductId'])), shape=(user_product ['MemberId'].max()+1, user_product['ProductId'].max()+1), dtype=np.uint16)



from sklearn.decomposition import TruncatedSVD
decomp = TruncatedSVD(n_components=10, random_state=101)
user_reduced = decomp.fit_transform(user_product_sparse)

#print(decomp.explained_variance_ratio_[:10], decomp.explained_variance_ratio_.sum())
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
user_reduced_scaled = scaler.fit_transform(user_reduced)

from sklearn.ensemble import IsolationForest
clf = IsolationForest(contamination=0.05, random_state=101)
clf.fit(user_reduced_scaled)
outliers = clf.predict(user_reduced_scaled)


unique, counts = np.unique(outliers, return_counts=True)
#print(dict(zip(unique, counts)))

user_product= user_product.drop(['DateOfBirth','ProfitCat','index'],axis=1)

#user_product=user_product.drop(user_product.columns[[1, 8]], axis=1, inplace=True)
from sklearn.decomposition import PCA
pca_reducer = PCA(n_components=2)
reduced_data = pca_reducer.fit_transform(user_product)
reduced_data.shape
from sklearn.cluster import KMeans
km = KMeans(n_clusters=4)
cluster = km.fit(reduced_data)


import numpy as np
import random

class KMeans_numpy(object):
    #INPUT: n_clusters - number of clusters that algortihm will try to find
    #       tolerance -  number when difference between prev cluster and new cluster is less then this number we will stop algo
    #       max_iter - how many times cetroids will move
    def __init__(self, n_clusters=4, tolerance = 0.001, max_iter = 300):
        self.k_clusters = n_clusters
        self.tol = tolerance
        self.max_iter = max_iter
    
    #TRAIN/FIT function, used to find the best positions for our clusters
    #
    #INPUT: X - fetures of dataset in which we are trying to find clusters
    def fit(self, X):
        #Starting clusters will be random members from X set
        self.centroids = []
        
        for i in range(self.k_clusters):
            #this index is used to acces random element from input set
            index = random.randint(1, len(X)-1)
            self.centroids.append(X[index])
        
        for i in range(self.max_iter):  
            #storing previous values of centroids
            prev_centroids = self.centroids
            #This will be dict for ploting data later on
            #with it we can find data points which are in the some cluster
            self.clustered_data = {}
            #Centroids values for this iteration
            cen_temp = []
            
            for centroid in range(len(self.centroids)):
                #creating empty list of elements for current cluster/centroid
                list_of_closer_samples = []
                
                for k in range(len(X)):
                    helper_list = []
                    for j in range(self.k_clusters):
                        #caluclating euclidian distance between current X value and all centroids in our list
                        helper_list.append(self.euclidian_distance(self.centroids[j], X[k]))
                    
                    #if minimal distance between curent value and centroid that we are currently interested in
                    #store value to the list of examples for that centroid
                    if np.argmin(helper_list) == centroid:
                        list_of_closer_samples.append(X[k])   
                
                #New position of each cluster is calculated by mean of all examples closest to it
                cen_temp.append(np.average(list_of_closer_samples, axis=0))
                
                self.clustered_data[centroid] = list_of_closer_samples

            #check if it is optimized
            optimized = True
            for c in range(len(self.centroids)):
                original_centroid = prev_centroids[c]
                current_centroid = cen_temp[c]
                #checking if % of change between old position and new position is less then tolerance (0.001 by default)
                if np.sum((current_centroid - original_centroid)/original_centroid*100.0) > self.tol:
                    optimized = False
                    self.centroids = cen_temp
                    
            if optimized:
                break
        #return final clusters values [optional, it is only used for graph]
        c = self.centroids
        return c, self.clustered_data
    
    #euclidian distance between points
    def euclidian_distance(self, centroid, x_current):
        return np.sqrt(np.sum(np.power((centroid - x_current), 2)))
    
    #INPUT: X_test set - set of data to test our clusters on
    def predict(self, Xtest):
        pred = []
        for i in range(len(Xtest)):
            help_list = []
            for j in range(len(self.centroids)):
                help_list.append(self.euclidian_distance(self.centroids[j], Xtest[i]))
            pred.append(np.argmin(help_list))
        print(pred)


km_numpy = KMeans_numpy(n_clusters=4, tolerance=0.0001)
clusters, clusterd_data = km_numpy.fit(reduced_data)
clusters = np.array(clusters)
cluster_one_data = np.array(clusterd_data[0])
cluster_two_data = np.array(clusterd_data[1])
cluster_three_data = np.array(clusterd_data[2])
cluster_four_data = np.array(clusterd_data[3])

#plt.figure(figsize=(12, 6))
#plt.scatter(cluster_one_data[:, 0], cluster_one_data[:, 1], c='r', label='Cluster One')
#plt.scatter(cluster_two_data[:, 0], cluster_two_data[:, 1], c='b', label='Cluster two')
#plt.scatter(cluster_three_data[:, 0], cluster_three_data[:, 1], c='c', label='Cluster three')
#plt.scatter(cluster_four_data[:, 0], cluster_four_data[:, 1], c='y', label='Cluster four')
#plt.scatter(clusters[:, 0], clusters[:, 1], marker='*', s=200, color='black', label='Centroids')
#plt.title("Custom KMeans results")
#plt.legend()
#plt.show()
full_data_kmeans = KMeans_numpy(n_clusters=4)
centroids, clus_data = full_data_kmeans.fit(user_product.values)

#g=user_product['Age'].as type(int)
cluster_1 = pd.DataFrame(clus_data[0], columns=['MemberId','prior_order_count','ProductId', 'Id', 'GenderId','Age','MaritalStatus',])
cluster_2 = pd.DataFrame(clus_data[1], columns=['MemberId','prior_order_count','ProductId', 'Id', 'GenderId', 'Age','MaritalStatus'])
cluster_3 = pd.DataFrame(clus_data[2], columns=['MemberId','prior_order_count','ProductId', 'Id', 'GenderId', 'Age','MaritalStatus'])
cluster_4 = pd.DataFrame(clus_data[3], columns=['MemberId','prior_order_count','ProductId', 'Id', 'GenderId', 'Age','MaritalStatus'])
#print(cluster_1)
#print("Cluster 1")
#print("////////////////////////////////////////")
#print("Average age for customers in cluster one: {}".format(np.array(cluster_1['Age']).mean()))
#print("Marital Status for customers in cluster one: {}".format(np.array(cluster_1['MaritalStatus']).mean()))
#print("In cluster one we have: {} customers".format(cluster_1.shape[0]))
#print("From those customers we have {} male and {} female".format(cluster_1.loc[(cluster_1['GenderId'] == 2.0)].shape[0], cluster_1.loc[(cluster_1['GenderId'] == 1.0)].shape[0]))
##print(cluster_2)
#print("Cluster 2")
#print("////////////////////////////////////////")
#print("Average age for customers in cluster two: {}".format(np.array(cluster_2['Age']).mean()))
#print("Marital Status   for customers in cluster two: {}".format(np.array(cluster_2['MaritalStatus']).mean()))
#print("In cluster two we have: {} customers".format(cluster_2.shape[0]))
#print("From those customers we have {} male and {} female".format(cluster_2.loc[(cluster_2['GenderId'] == 2.0)].shape[0], cluster_2.loc[(cluster_2['GenderId'] == 1.0)].shape[0]))
##print(cluster_3)
#print("Cluster 3")
#print("////////////////////////////////////////")
#print("Average age for customers in cluster three: {}".format(np.array(cluster_3['Age']).mean()))
#print("Marital Status for customers in cluster three: {}".format(np.array(cluster_3['MaritalStatus']).mean()))
#print("In cluster three we have: {} customers".format(cluster_3.shape[0]))
#print("From those customers we have {} male and {} female".format(cluster_3.loc[(cluster_3['GenderId'] == 2.0)].shape[0], cluster_3.loc[(cluster_3['GenderId'] == 1.0)].shape[0]))
#
##print(cluster_4)
#print("Cluster 4")
#print("////////////////////////////////////////")
#print("Average age for customers in cluster four: {}".format(np.array(cluster_4['Age']).mean()))
#print("Marital Status for customers in cluster  four: {}".format(np.array(cluster_4['MaritalStatus']).mean()))
#print("In cluster four we have: {} customers".format(cluster_4.shape[0]))
#print("From those customers we have {} male and {} female".format(cluster_4.loc[(cluster_4['GenderId'] == 2.0)].shape[0], cluster_4.loc[(cluster_4['GenderId'] == 1.0)].shape[0]))

import matplotlib.pyplot as plt

# red is an outlier, green is a regular observation
color_map = np.vectorize({ -1: 'r', 1: 'b'}.get)

#plt.scatter(user_reduced_scaled[:,0], user_reduced_scaled[:,1], c=color_map(outliers), alpha=0.1)
from sklearn.cluster import KMeans
clusters_count = 10

kmc = KMeans(n_clusters=clusters_count, init='random', n_init=10, random_state=101)
kmc.fit(user_reduced_scaled[outliers==-1,:])
clusters = kmc.predict(user_reduced_scaled)

unique, counts = np.unique(clusters, return_counts=True)
#print("Counts:",dict(zip(unique, counts)))
#plt.scatter(user_reduced_scaled[:,0], user_reduced_scaled[:,1], c=clusters / (clusters_count-1), cmap='tab10', alpha=0.1)

top_products_overall =user_product[['ProductId','prior_order_count']].groupby('ProductId').sum().reset_index().sort_values('prior_order_count', ascending=False)

top_products_overall['rank_overall'] = top_products_overall['prior_order_count'].rank(ascending=False)

# packing clusters we found into dataframe
usersdf = pd.DataFrame(clusters[1:], columns=['cluster'], index=np.arange(1, user_product ['MemberId'].max()+1))

top_products = user_product .merge(usersdf, left_on='MemberId', right_index=True)
top_products['rank'] = top_products[['cluster','prior_order_count']].groupby('cluster').rank(ascending=False)
#user_product=pd.merge(pd.merge(orders ,item_prior,on='MemberId'),customers_data,on='MemberId')
# merging with overall top products
top_products = top_products.merge(top_products_overall[['ProductId','rank_overall']], left_on='ProductId',right_on='ProductId')

# calculating differences between ranks
top_products['rank_diff'] = top_products['rank'] - top_products['rank_overall']
# leaving top products in each cluster: 2 with largest and 2 with smallest difference in ranks
top_products_asc_diff = top_products.sort_values(['cluster','rank_diff'], ascending=False).groupby('cluster').head(2).reset_index(drop=True)
#print(top_products_asc_diff)
top_products_desc_diff = top_products.sort_values(['cluster','rank_diff'], ascending=True).groupby('cluster').head(2).reset_index(drop=True)
#print(top_products_desc_diff)
top_products_diff = pd.concat([top_products_asc_diff,top_products_desc_diff], axis=0)
# printing results
c=top_products_diff.merge(products[['ProductName']], left_on='ProductId', right_index=True)[['cluster','ProductName','rank','MemberId','rank_overall']].sort_values(['cluster','rank'])

d=pd.DataFrame.drop_duplicates(c)
#print (d)
#userItemData = pd.read_csv('order_products_prior1.csv')
item_prior.head(2000)
#Get list of unique items
itemList=list(set(item_prior["ProductId"].tolist()))


#Get count of users
userCount=len(set(item_prior["ProductId"].tolist()))
#Create an empty data frame to store it affinity scores for items.
#print("\n")
#print("Item-Item Recommendation")
itemAffinity= pd.DataFrame(columns=('Product1', 'Product2', 'score'))


rowCount=0

#For each item in the list, compare with other items.
for ind1 in range(len(itemList)):
    
    
    #Get list of users who bought this item 1.
    item1Users = item_prior[item_prior.ProductId==itemList[ind1]]["MemberId"].tolist()
#    print("Item 1 ", item1Users)
    
    #Get item 2 - items that are not item 1 or those that are not analyzed already.
    for ind2 in range(ind1, len(itemList)):
        
        if ( ind1 == ind2):
            continue
  #Get list of users who bought item 2
        item2Users=item_prior[item_prior.ProductId==itemList[ind2]]["MemberId"].tolist()
#        print("Item 2",item2Users)
        
                
#Find score. Find the common list of users and divide it by the total users.
        commonUsers= len(set(item1Users).intersection(set(item2Users)))
        score=commonUsers / userCount
        
                #Add a score for item 1, item 2
        itemAffinity.loc[rowCount] = [itemList[ind1],itemList[ind2],score]
        rowCount +=1
                #Add a score for item2, item 1. The same score would apply irrespective of the sequence.
        itemAffinity.loc[rowCount] = [itemList[ind2],itemList[ind1],score]
        rowCount +=1
        
        
       
#Check final result
print(itemAffinity.head(2000))
#searchItem=6150

searchItem=int(input("Enter the id of search item"))

recoList=itemAffinity[itemAffinity.Product1==searchItem] \
       [["Product2","score"]] .sort_values("score", ascending=[0])
       
#print("Recommendations for the search item ",+searchItem ,"\n ", recoList)

print("Items similar to your search item ")
print("Recommended for you")
#print("Products","\t","\t","score")
list=[]
i=1
for row in recoList.itertuples():
   if row.score>0.01:
#        print(row.Product2,"\t","\t","\t",row.score)
        dict={}
       
        dict['Product'+str(i)]=row.Product2
        dict['score']=row.score
        i+=1
        list.append(dict)
print(list)
       
#
#from sklearn.metrics import mean_absolute_error
#print(mean_absolute_error(recoList,recoList))
        
        

        

