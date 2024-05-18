import pandas as pd
import pickle
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB



def train_naive_bayes_model(data_processed):
    # Combine text and topic for feature extraction
    X = data_processed['Text'] + data_processed["Topic"]
    y = data_processed['Sentiment (Label)']

    # Initialize CountVectorizer
    featureExtraction_BagOfword = CountVectorizer(min_df=1, stop_words="english", lowercase=True)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1007)

    # Transform the text data to bag-of-words features
    X_train_featureExtraction_BagOfword = featureExtraction_BagOfword.fit_transform(X_train)
    X_test_featureExtraction_BagOfword = featureExtraction_BagOfword.transform(X_test)

    # Train the Naive Bayes model
    naive_bayes_model_bow = MultinomialNB()
    naive_bayes_model_bow.fit(X_train_featureExtraction_BagOfword, y_train)

    # Predict on the test set
    y_prediction = naive_bayes_model_bow.predict(X_test_featureExtraction_BagOfword)

    # Calculate and print the accuracy
    accuracy = 100 * accuracy_score(y_test, y_prediction)
    print("Naive Bayes accuracy: ", accuracy)

    # Perform cross-validation
    cross_val_scores = cross_val_score(naive_bayes_model_bow, featureExtraction_BagOfword.transform(X), y, cv=5)
    cross_val_mean = 100 * cross_val_scores.mean()
    print("Naive Bayes cross-validation accuracy: ", cross_val_mean)

    # Save the model and vectorizer
    with open('naive_bayes_model.pkl', 'wb') as model_file:
        pickle.dump(naive_bayes_model_bow, model_file)
    with open('count_vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(featureExtraction_BagOfword, vectorizer_file)
        


def predict_sentiment_naive_bayes(text):
    # Preprocess the input text
    text_preprocessed = text

    # Load the saved model
    with open('naive_bayes_model.pkl', 'rb') as model_file:
        naive_bayes_model_bow = pickle.load(model_file)

    # Load the saved CountVectorizer
    with open('count_vectorizer.pkl', 'rb') as vectorizer_file:
        featureExtraction_BagOfword = pickle.load(vectorizer_file)

    # Vectorize the input text
    text_features = featureExtraction_BagOfword.transform([text_preprocessed])

    # Predict the sentiment
    sentiment_prediction = naive_bayes_model_bow.predict(text_features)

    return sentiment_prediction[0]