import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

def train_LogisticRegression_model(data_processed):
    # Combine text and topic for feature extraction
    X = data_processed['Text'] + data_processed["Topic"]
    y = data_processed['Sentiment (Label)']

    # Initialize TFIDF vectorizer
    featureExtraction_TFIDF = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Transform the text data to TFIDF features
    X_train_featureExtraction_TFIDF = featureExtraction_TFIDF.fit_transform(X_train)
    X_test_featureExtraction_TFIDF = featureExtraction_TFIDF.transform(X_test)

    # Train the Logistic Regression model
    logisticRegModel = LogisticRegression(solver='liblinear', C=10.0, random_state=0).fit(X_train_featureExtraction_TFIDF, y_train)

    # Predict on the test set
    y_prediction = logisticRegModel.predict(X_test_featureExtraction_TFIDF)

    # Calculate and print the accuracy
    accuracy = 100 * accuracy_score(y_test, y_prediction)
    print("Logistic Regression accuracy: ", accuracy)

    # Perform cross-validation
    cross_val_scores = cross_val_score(logisticRegModel, featureExtraction_TFIDF.transform(X), y, cv=5)
    cross_val_mean = 100 * cross_val_scores.mean()
    print("Logistic Regression cross-validation accuracy: ", cross_val_mean)

    # Save the model and vectorizer
    with open('logistic_model.pkl', 'wb') as model_file:
        pickle.dump(logisticRegModel, model_file)
    with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(featureExtraction_TFIDF, vectorizer_file)
        

def predict_sentiment_LogisticRegression(text):
    # Load the saved model
    with open('logistic_model.pkl', 'rb') as model_file:
        logisticRegModel = pickle.load(model_file)

    # Load the saved TFIDF vectorizer
    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        featureExtraction_TFIDF = pickle.load(vectorizer_file)

    # Vectorize the input text
    text_features = featureExtraction_TFIDF.transform([text])

    # Predict the sentiment
    sentiment_prediction = logisticRegModel.predict(text_features)

    return sentiment_prediction[0]