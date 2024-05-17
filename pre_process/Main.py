import pandas as pd
from remove_stop_words import remove_stop_words
from text_normalization import Converting_To_Primitive
from convert_to_lowercase import convert_to_lowercase
from remove_extra_spaces import remove_extra_spaces
from remove_punctuation import remove_punctuation
from remove_slangs import remove_slangs
from lemmatization_text import lemmatization_text
from SVM_Model import predict_sentiment_SVM
from LogisticRegression_Model import predict_sentiment_LogisticRegression
from Naive_Bayes_Model import predict_sentiment_naive_bayes

# Read data
data = pd.read_csv('../sentimentdataset.csv')

def data_preprocess(data_to_process):
    # Load the data and preprocess

    data_to_process = remove_extra_spaces(data_to_process)
    data_to_process = convert_to_lowercase(data_to_process, 'Text')
    data_to_process = convert_to_lowercase(data_to_process, 'Sentiment (Label)')
    data_to_process['Text'] = data_to_process['Text'].apply(remove_punctuation)
    data_to_process['Text'] = data_to_process['Text'].apply(remove_slangs)
    data_to_process['Sentiment (Label)'] = data_to_process['Sentiment (Label)'].apply(Converting_To_Primitive)
    data_to_process['Text'] = data_to_process['Text'].apply(lemmatization_text)

    return data_to_process

def text_preprocess(text):
    # preprocess Text
    text = text.lower()
    text = remove_stop_words(text)
    text = remove_punctuation(text)
    text = remove_slangs(text)
    text = lemmatization_text(text)

    return text

# Test the function
data = data_preprocess(data)

while True:
    text_to_predict = input("enter a sentence: ")
    text_to_predict = text_preprocess(text_to_predict)
    predicted_sentiment = predict_sentiment_SVM(text_to_predict, data)
    print("Predicted sentiment from SVM:", predicted_sentiment)

    predicted_sentiment = predict_sentiment_LogisticRegression(text_to_predict, data)
    print("Predicted sentiment from Logistic Regression:", predicted_sentiment)

    predicted_sentiment = predict_sentiment_naive_bayes(text_to_predict, data)
    print("Predicted sentiment from Naive Bayes:", predicted_sentiment)
    print()
    next_input = input("Do you want to enter another sentence?(y/n) ")
    if next_input == 'n':
        break