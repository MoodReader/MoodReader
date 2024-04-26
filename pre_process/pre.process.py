import pandas as pd
from convert_to_lowercase import convert_to_lowercase
from remove_stop_words import remove_stop_words

data = pd.read_csv('../sentimentdataset.csv')

data = convert_to_lowercase(data, 'Text')
print(data['Text'].head())

data['Text'] = data['Text'].apply(remove_stop_words)
print(data['Text'].head())
