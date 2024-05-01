import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import nltk 
from nltk.corpus import stopwords
import string
from text_normalization import*
from remove_slangs import*

data = pd.read_csv('sentimentdataset.csv')

for row in data.index:
   data.loc[row,"Text"] = data.loc[row,"Text"].lower()

for row in data.index:
   data.loc[row,"Sentiment (Label)"] = data.loc[row,"Sentiment (Label)"].lower()
   
# Removing punctuation and stop words 
punctuation = list(string.punctuation)
for i in punctuation:
    data["Text"] = data["Text"].str.replace(i, " ") 

stop_words= stopwords.words('english')+ ["❤️" , "🐾", "🐶","💪","🎉","🎨","🎂"]
for i in stop_words:
    data["Text"] = data["Text"].str.replace(' '+i+' ', " ") 

# remove extra spaces
for row in data.index:
   data.loc[row,"Text"] = " ".join(data.loc[row,"Text"].split())

data["Sentiment (Label)"] = data["Sentiment (Label)"].apply(remove_slangs)

data["Sentiment (Label)"] = data["Sentiment (Label)"].apply(Converting_To_Primitive)

print(data["Text"])