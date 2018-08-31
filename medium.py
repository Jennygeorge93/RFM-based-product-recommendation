#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 15:32:17 2018

@author: user
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
data=pd.read_csv('RFM_Full.csv')
#print(data.head())
data['Total_Price']=data['Quantity']*data['UnitPrice']
data['date']=data['InvoiceDate'].str.extract('(.*)-').str.extract('(.*)-')
data['date']=data.date.astype(str).str.zfill(2)
data['date']=data['date'] + data['InvoiceDate'].str.extract('-(.*) ').str.extract('(.*)-') 
data.date = pd.to_numeric(data.date, errors='coerce')
Cust_country=data[['City','MemberId']].drop_duplicates()
Cust_country_count=Cust_country.groupby(['City'])['MemberId'].\
aggregate('count').reset_index().sort_values(by='MemberId', ascending=False)
country=list(Cust_country_count['City'])
Cust_id=list(Cust_country_count['MemberId'])
plt.figure(figsize=(12,8))
sns.barplot(country, Cust_id, alpha=0.8, color=color[4])
plt.xticks(rotation='60')
plt.show()
Cust_date_UK=data[data['City']=="New York City"]
Cust_date_UK=Cust_date_UK[['MemberId','date']].drop_duplicates()
def f(row):
    
    if row['date'] > 201710:
        val = 1
    elif row['date'] <= 201710 and row['date'] > 201701:
        val = 2
    elif row['date'] <=201701  and row['date'] >201608 :
        val = 3
    elif row['date'] <=201608   and row['date'] >201506:
        val = 4
    else:
        val = 5
    return val
Cust_date_UK['Recency_Flag'] = Cust_date_UK.apply(f, axis=1)
Cust_date_UK = Cust_date_UK.groupby("MemberId",as_index=False)["Recency_Flag"].max()
plt.figure(figsize=(8,6))
sns.countplot(x="Recency_Flag", data=Cust_date_UK, color=color[2])
plt.ylabel('Count', fontsize=12)
plt.xlabel('Recency_Flag', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency of Recency_Flag", fontsize=15)
plt.show()

Cust_monetary = data.groupby(['City','MemberId'])['Total_Price'].aggregate("sum").\
reset_index().sort_values(by='Total_Price', ascending=False)
Cust_monetary_UK=Cust_monetary[Cust_monetary['City']=="New York City"]
#unique_price=Cust_monetary_UK[['Total_Price']].drop_duplicates()
#unique_price=unique_price[unique_price['Total_Price'] > 0]
#unique_price['monetary_Band'] = pd.qcut(unique_price['Total_Price'], 5)
#unique_price=unique_price[['monetary_Band']].drop_duplicates()
#unique_price
def f(row):
    if row['Total_Price'] <= 1000:
        val = 5
    elif row['Total_Price'] > 1000 and row['Total_Price'] <= 2000:
        val = 4
    elif row['Total_Price'] > 2000 and row['Total_Price'] <= 3000:
        val = 3
    elif row['Total_Price'] > 3000 and row['Total_Price'] <= 4000:
        val = 2
    else:
        val = 1
    return val
Cust_monetary_UK['Monetary_Flag'] = Cust_monetary_UK.apply(f, axis=1)
#Let us check the distribution of Monetary flags:

plt.figure(figsize=(8,6))
sns.countplot(x="Monetary_Flag",data=Cust_monetary_UK, color=color[3])
plt.ylabel('Count', fontsize=12)
plt.xlabel('Monetary_Flag')
plt.title("Frequency of Monetary_Flag", fontsize=15)
plt.show()

Cust_freq=data[['City','InvoiceNo','MemberId']].drop_duplicates()


#Cust_freq_count=Cust_freq.groupby(["City","MemberId"])["InvoiceNo"].aggregate("count").\
#reset_index().sort_values(by='InvoiceNo', ascending=False)
#print(Cust_freq_count)
#Cust_freq_count_UK=Cust_freq_count[Cust_freq_count['City']=="New York City"]
Cust_freq_count_UK=Cust_freq[Cust_freq['City']=="New York City"]


def f(row):
   
    if int(row['InvoiceNo']) <= 536367:
        val = 5
    elif int(row['InvoiceNo']) > 536368 and int(row['InvoiceNo']) <= 536375:
        val = 4
    elif int(row['InvoiceNo'])> 536376 and int(row['InvoiceNo'])<= 536380:
        val = 3
    elif int(row['InvoiceNo']) > 536381 and int(row['InvoiceNo'])<= 536420:
        val = 2

    else:
        val = 1
    return val


Cust_freq_count_UK['Freq_Flag'] = Cust_freq_count_UK.apply(f, axis=1)
plt.figure(figsize=(8,6))
sns.countplot(x="Freq_Flag", data=Cust_freq_count_UK, color=color[1])
plt.ylabel('Count', fontsize=12)
plt.xlabel('Freq_Flag', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency of Freq_Flag", fontsize=15)
plt.show()
##
#
#
#
#
#Cust_UK_All=pd.merge(Cust_date_UK,Cust_freq_count_UK[['MemberId','Freq_Flag']],\
#on=['MemberId'],how='left')
#Cust_UK_All=pd.merge(Cust_UK_All,Cust_monetary_UK[['MemberId','Monetary_Flag']],\
#on=['MemberId'],how='left')
#print(Cust_UK_All.head(1000))
