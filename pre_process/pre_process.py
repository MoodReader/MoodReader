import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import nltk 
from nltk.corpus import stopwords
import string

data = pd.read_csv('sentimentdataset.csv')

for row in data.index:
   data.loc[row,"Text"] = data.loc[row,"Text"].lower()

# Removing punctuation and stop words
punctuation = list(string.punctuation) + ["\" "," \""," \" "]
for i in punctuation:
    data["Text"] = data["Text"].str.replace(i, " ") 

stop_words= stopwords.words('english')+ ["❤️" , "🐾", "🐶","💪","🎉","🎨","🎂"]
for i in stop_words:
    data["Text"] = data["Text"].str.replace(' '+i+' ', " ") 
    
# remove extra spaces
for row in data.index:
   data.loc[row,"Text"] = " ".join(data.loc[row,"Text"].split())


print(data["Text"].head())
