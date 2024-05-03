import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from text_normalization import Converting_To_Primitive
from convert_to_lowercase import convert_to_lowercase
from remove_extra_spaces import remove_extra_spaces
from remove_punctuation import remove_punctuation
from remove_slangs import remove_slangs
from lemmatization_text import lemmatization_text

data = pd.read_csv('../sentimentdataset.csv')

data = remove_extra_spaces(data)
data = convert_to_lowercase(data, 'Text')
data = convert_to_lowercase(data, 'Sentiment (Label)')
data['Text'] = data['Text'].apply(remove_punctuation)
data['Text'] = data['Text'].apply(remove_slangs)
data['Sentiment (Label)'] = data['Sentiment (Label)'].apply(Converting_To_Primitive)
data['Text'] = data['Text'].apply(lemmatization_text)

X = data['Text'] + data["Topic"]
y = data['Sentiment (Label)']

featureExtraction_TFIDF = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)
featureExtraction_BagOfword = CountVectorizer(min_df=1, stop_words="english", lowercase=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train_featureExtraction_TFIDF = featureExtraction_TFIDF.fit_transform(X_train)# is for SVM and Logistic Regression
X_test_featureExtraction_TFIDF = featureExtraction_TFIDF.transform(X_test)

X_train_featureExtraction_BagOfword = featureExtraction_BagOfword.fit_transform(X_train)# is for Naive Bayes
X_test_featureExtraction_BagOfword = featureExtraction_BagOfword.transform(X_test)

logisticRegModel = LogisticRegression(solver='liblinear', C=10.0, random_state=0).fit(X_train_featureExtraction_TFIDF, y_train)
y_pred1 = logisticRegModel.predict(X_test_featureExtraction_TFIDF)

print('The accuracy for Logistic Regression Classifier:', accuracy_score(y_test, y_pred1) * 100)

conf_m = confusion_matrix(y_test, y_pred1)
report = classification_report(y_test, y_pred1)
print('report: ', report, sep='\n')
print('-----------------------------------------------------')
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred1)).plot()
plt.title("Confusion Matrix for Logistic Regression Classifier")

svmModel = SVC(kernel='linear').fit(X_train_featureExtraction_TFIDF, y_train)
y_pred2 = svmModel.predict(X_test_featureExtraction_TFIDF)

print('The accuracy for Support Vector Machines (SVM):', accuracy_score(y_test, y_pred2) * 100)

conf_m = confusion_matrix(y_test, y_pred2)
report = classification_report(y_test, y_pred2)
print('report: ', report, sep='\n')
print('-----------------------------------------------------')
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred2)).plot()
plt.title("Confusion Matrix for Support Vector Machines (SVM)")

naive_bayes_model_bow = MultinomialNB()
naive_bayes_model_bow.fit(X_train_featureExtraction_BagOfword, y_train)
y_pred_nb_bow = naive_bayes_model_bow.predict(X_test_featureExtraction_BagOfword)
print('The accuracy for Naive Bayes (Bag of Words):', accuracy_score(y_test, y_pred_nb_bow) * 100)

conf_m = confusion_matrix(y_test, y_pred_nb_bow)
report = classification_report(y_test, y_pred_nb_bow)
print('report: ', report, sep='\n')
print('-----------------------------------------------------')
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_nb_bow)).plot()
plt.title("Confusion Matrix for Naive Bayes (Bag of Words)")


# # the faild LSTM or RNN
# from sklearn.preprocessing import MaxAbsScaler  
# scaler = MinMaxScaler()
# scaler.fit(X_train_featureExtraction_TFIDF)
# scaled_train = scaler.transform(X_train_featureExtraction_TFIDF)
# scaled_test = scaler.transform(X_test_featureExtraction_BagOfword)

# from keras.preprocessing.sequence import TimeseriesGenerator
 
# n_input = 3
# n_features = 1
# generator = TimeseriesGenerator(scaled_train,
#                                 scaled_train,
#                                 length=n_input,
#                                 batch_size=1)
# X, y = generator[0]
# print(f'Given the Array: \n{X.flatten()}')
# print(f'Predict this y: \n {y}')
# # We do the same thing, but now instead for 12 months
# n_input = 12
# generator = TimeseriesGenerator(scaled_train,
#                                 scaled_train,
#                                 length=n_input,
#                                 batch_size=1)

# model = Sequential()
# model.add(LSTM(100, activation='relu',
#                input_shape=(n_input, n_features)))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')
# model.summary()
# model.fit(generator, epochs=5)

# # Cross validation
# models = [
#     ('LogisticRegression', LogisticRegression()),
#     ('SVC', svm.SVC(kernel='linear')),
#     ('NaiveBayesClassifier', MultinomialNB()),
# ]

# kf = KFold(n_splits=5)

# for name, model in models:
#     pipe = Pipeline([
#         ('feature_extraction', TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)),
#         ('model', model)
#     ])
#     scores = cross_val_score(pipe, X, y, cv=kf)
#     accuracy = np.mean(scores)
#     print(f"{name} Accuracy: {accuracy*100:.2f}%")