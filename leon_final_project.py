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
