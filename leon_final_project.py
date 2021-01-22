#IMPORTS

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

#--------------GETTING THE DATA---------------

data = pd.read_csv("Country-data.csv")


#----------------PREPROCESSING---------------

print("---Checking the duplicated and null values---")

duplicate_check=sum(data.duplicated(subset = 'country')) == 0
print("True if there is no duplicated value:",duplicate_check)

print("--------------------------")

is_null=data.isnull().sum()
print("0 if there is no null value:")
print(is_null)

#According to factors, plotting the 10 countries that has the worst conditions

import matplotlib.pyplot as plt
import seaborn as sns

fig, axs = plt.subplots(3,3,figsize = (15,15))

columns_False=['child_mort','total_fer','inflation']
columns_True=['life_expec','health','gdpp','income','exports','imports']

axsFalse=[axs[0,0],axs[0,1],axs[0,2]]
axsTrue=[axs[1,0],axs[1,1],axs[1,2],axs[2,0],axs[2,1],axs[2,2]]

for column in range(3):
    for ax in axs:
        top10 = data[['country',columns_False[column]]].sort_values(columns_False[column], ascending = False).head(10)
        plt1 = sns.barplot(x='country', y=columns_False[column], data= top10, ax = axsFalse[column],palette="ch:s=-.2")
        plt1.set(xlabel = '', ylabel= columns_False[column])

for column in range(6):
    for ax in axs:
        top10 = data[['country',columns_True[column]]].sort_values(columns_True[column], ascending = True).head(10)
        plt1 = sns.barplot(x='country', y=columns_True[column], data= top10, ax = axsTrue[column],palette="ch:s=-.2")
        plt1.set(xlabel = '', ylabel= columns_True[column])
            
for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation = 90)
    
plt.tight_layout()
plt.savefig("worst_condition.pdf")
plt.close()


#-------------Plotting the Heatmap to see the correlations between variables---------------

plt.figure(figsize = (8,6))  
sns.heatmap(data.corr(),annot = True,cmap="Purples")
plt.savefig("heatmap.pdf")
plt.close()

#----------------Making import export and healt values looking like gdpp values, not percentage-----------------
for i in ['imports','exports','health']:
    data[i] = (data[i] * data['gdpp'])/100
    
    
#Standard scaler

from sklearn.preprocessing import StandardScaler

new_df = data[data.columns[data.dtypes != 'object']]

scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(new_df))

#-----------------CLUSTERING TENDENCY---------------------

from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
from math import isnan


def hopkins(X):
    d = X.shape[1]
    n = len(X) 
    m = int(0.1 * n) 
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
 
    rand_X = sample(range(0, n, 1), m)
 
    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])
 
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print(ujd, wjd)
        H = 0
 
    return H


print("------------------------")
print("Hopkins score:",hopkins(new_df))
print("------------------------")

#---------------Finding the optimum number of Clusters - Elbow Curve-------------------------

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

inertia_list = []
cluster_labels=[]

num_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_cluster in num_clusters:
    kmeans = KMeans(n_clusters=num_cluster,max_iter=50)
    kmeans.fit(data_scaled)
    
    inertia_list.append(kmeans.inertia_)
    cluster_labels = kmeans.labels_
    silhouette_avg = silhouette_score(data_scaled, cluster_labels)
    print("For n_clusters={0}, the silhouette score is {1}".format(num_cluster, silhouette_avg))

df_inertia = pd.DataFrame(inertia_list)
df_inertia.columns = ['Inertia']

plt.plot(num_clusters, df_inertia['Inertia'], 'bx-',color='purple')

plt.title('Inertia/Elbow Curve')
plt.savefig('inertia.pdf')
plt.close()

print("--------------------------")

#------------------------------kMeans with 3 cluster---------------------

kmeans_3cluster = KMeans(n_clusters=3, init='k-means++', random_state=42).fit(data_scaled)
data['kmeans_3cluster_labels'] = kmeans_3cluster.labels_


sns.scatterplot(x='income', y='child_mort', hue='kmeans_3cluster_labels',data=data, legend='full', palette="flare",s=100,alpha=0.7)
plt.savefig('3cluster.pdf')
plt.close()

print("--------------------------")
