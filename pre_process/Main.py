import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from remove_stop_words import remove_stop_words
from text_normalization import Converting_To_Primitive
from convert_to_lowercase import convert_to_lowercase
from remove_extra_spaces import remove_extra_spaces
from remove_punctuation import remove_punctuation
from remove_slangs import remove_slangs
from lemmatization_text import lemmatization_text

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


def predict_sentiment_SVM(text, data_processed):

    X = data_processed['Text'] + data_processed["Topic"]
    y = data_processed['Sentiment (Label)']

    featureExtraction_TFIDF = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    X_train_featureExtraction_TFIDF = featureExtraction_TFIDF.fit_transform(X_train)  # is for SVM and Logistic Regression
    X_test_featureExtraction_TFIDF = featureExtraction_TFIDF.transform((X_test))

    # Train the model
    svmModel = SVC(kernel='linear').fit(X_train_featureExtraction_TFIDF, y_train)

    # Vectorize the input text
    text_features = featureExtraction_TFIDF.transform([text])

    y_prediction = svmModel.predict(X_test_featureExtraction_TFIDF)

    # Predict the sentiment
    sentiment_prediction = svmModel.predict(text_features)

    print("SVM accuracy: ", 100*accuracy_score(y_test,y_prediction))

    return sentiment_prediction[0]


def predict_sentiment_LogisticRegression(text, data_processed):

    X = data_processed['Text'] + data_processed["Topic"]
    y = data_processed['Sentiment (Label)']

    featureExtraction_TFIDF = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    X_train_featureExtraction_TFIDF = featureExtraction_TFIDF.fit_transform(X_train)  # is for SVM and Logistic Regression
    X_test_featureExtraction_TFIDF = featureExtraction_TFIDF.transform((X_test))

    # Train the model
    logisticRegModel = LogisticRegression(solver='liblinear', C=10.0, random_state=0).fit(X_train_featureExtraction_TFIDF, y_train)

    # Vectorize the input text
    text_features = featureExtraction_TFIDF.transform([text])

    y_prediction = logisticRegModel.predict(X_test_featureExtraction_TFIDF)

    # Predict the sentiment
    sentiment_prediction = logisticRegModel.predict(text_features)

    print("logistic accuracy: ", 100*accuracy_score(y_test, y_prediction))

    return sentiment_prediction[0]


def predict_sentiment_naive_bayes(text, data_processed):

    X = data_processed['Text'] + data_processed["Topic"]
    y = data_processed['Sentiment (Label)']

    featureExtraction_BagOfword = CountVectorizer(min_df=1, stop_words="english", lowercase=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    X_train_featureExtraction_BagOfword = featureExtraction_BagOfword.fit_transform(X_train)  # is for Naive Bayes
    X_test_featureExtraction_BagOfword = featureExtraction_BagOfword.transform(X_test)

    # Train the model
    naive_bayes_model_bow = MultinomialNB()
    naive_bayes_model_bow.fit(X_train_featureExtraction_BagOfword, y_train)

    # Vectorize the input text
    text_features = featureExtraction_BagOfword.transform([text])

    # Predict the sentiment
    sentiment_prediction = naive_bayes_model_bow.predict(text_features)

    y_prediction = naive_bayes_model_bow.predict(X_test_featureExtraction_BagOfword)

    print("Naive Bayes accuracy: ", 100 * accuracy_score(y_test, y_prediction))
    return sentiment_prediction[0]


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
