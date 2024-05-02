import pandas as pd
import nltk
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('punkt')
from convert_to_lowercase import convert_to_lowercase
from remove_stop_words import remove_stop_words
from remove_punctuation import remove_punctuation
from remove_slangs import remove_slangs
from remove_extra_spaces import remove_extra_spaces
from lemmatization_text import lemmatization_text
from text_normalization import Converting_To_Primitive
data = pd.read_csv('../sentimentdataset.csv')

data = remove_extra_spaces(data)

data = convert_to_lowercase(data, 'Text')

data['Text'] = data['Text'].apply(remove_punctuation)

data['Text'] = data['Text'].apply(remove_stop_words)

data['Text'] = data['Text'].apply(remove_slangs)

data['Sentiment (Label)'] = data['Sentiment (Label)'].apply(Converting_To_Primitive)

data['Text'] = data['Text'].apply(lemmatization_text)
print(data['Text'].head())
# text = "Just published a new blog post. Check it out!"

# text = remove_punctuation(text)
# print(text)