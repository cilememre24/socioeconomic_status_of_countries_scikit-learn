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
