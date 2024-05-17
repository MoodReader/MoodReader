from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.svm import SVC

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