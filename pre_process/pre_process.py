import numpy as np
import pandas as pd
import nltk
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn import tree, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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

from sklearn.metrics import classification_report,confusion_matrix
import gensim
from gensim.models import Word2Vec,KeyedVectors
import pandas as pd
import nltk

data = pd.read_csv('../sentimentdataset.csv')

data = remove_extra_spaces(data)

data = convert_to_lowercase(data, 'Text')

data = convert_to_lowercase(data, 'Sentiment (Label)')

data['Text'] = data['Text'].apply(remove_punctuation)

data['Text'] = data['Text'].apply(remove_stop_words)

data['Text'] = data['Text'].apply(remove_slangs)

data['Sentiment (Label)'] = data['Sentiment (Label)'].apply(Converting_To_Primitive)

data['Text'] = data['Text'].apply(lemmatization_text)

X = data['Text']

y = data['Sentiment (Label)']

featureExtraction_TFIDF = TfidfVectorizer(min_df= 1, stop_words= "english", lowercase=True)#TF-IDF is for SVM

featureExtraction_BagOfword = CountVectorizer(min_df=1, stop_words="english", lowercase=True)#Bag of word is for Logistic Regression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train_featureExtraction_TFIDF = featureExtraction_TFIDF.fit_transform(X_train)
X_test_featureExtraction_TFIDF = featureExtraction_TFIDF.transform(X_test)

X_train_featureExtraction_BagOfword = featureExtraction_BagOfword.fit_transform(X_train)
X_test_featureExtraction_BagOfword = featureExtraction_BagOfword.transform(X_test)


logisticRegModel = LogisticRegression(solver='liblinear',C=10.0,random_state=42).fit(X_train_featureExtraction_BagOfword, y_train)
y_pred1 = logisticRegModel.predict(X_test_featureExtraction_BagOfword)


print('The accuracy for Logistic Regression Classifer:',accuracy_score(y_test,y_pred1)*100)

# conf_m= confusion_matrix(y_test,y_pred1)
# report = classification_report(y_test,y_pred1)
# print('report: ',report,sep='\n')

svmModel = svm.SVC(kernel='linear').fit(X_train_featureExtraction_TFIDF, y_train)
y_pred2 = svmModel.predict(X_test_featureExtraction_TFIDF)

print('The accuracy for Support Vector Machines (SVM):',accuracy_score(y_test,y_pred2)*100)

# conf_m= confusion_matrix(y_test,y_pred2)
# report = classification_report(y_test,y_pred2)
# print('report: ',report,sep='\n')

# text=X.values

# textvec=[nltk.word_tokenize(Text) for Text in text]

# #print(textvec)
# model1 = gensim.models.Word2Vec(textvec, min_count=5,window=100)

# model2 = gensim.models.Word2Vec(textvec, min_count=4,window=100, sg=1)

# print(model1.wv.most_similar('kindness'))
# print(model2.wv.most_similar('kindness'))


from sklearn.naive_bayes import MultinomialNB

# Naive Bayes using Bag of Words features
naive_bayes_model_bow = MultinomialNB()
naive_bayes_model_bow.fit(X_train_featureExtraction_BagOfword, y_train)
y_pred_nb_bow = naive_bayes_model_bow.predict(X_test_featureExtraction_BagOfword)
print('The accuracy for Naive Bayes (Bag of Words):', accuracy_score(y_test, y_pred_nb_bow) * 100)

