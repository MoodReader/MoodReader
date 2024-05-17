from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB



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