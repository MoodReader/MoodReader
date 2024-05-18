# This Python file uses the following encoding: utf-8
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from text_normalization import Converting_To_Primitive
from convert_to_lowercase import convert_to_lowercase
from remove_extra_spaces import remove_extra_spaces
from remove_punctuation import remove_punctuation
from remove_slangs import remove_slangs
from lemmatization_text import lemmatization_text
from remove_stop_words import remove_stop_words
def pre_process(data):
    data = remove_extra_spaces(data)
    data = convert_to_lowercase(data, 'Text')
    data = convert_to_lowercase(data, 'Sentiment (Label)')
    data['Text'] = data['Text'].apply(remove_punctuation)
    data ['Text'] = data['Text'].apply(remove_stop_words)
    data['Text'] = data['Text'].apply(remove_slangs)
    data['Sentiment (Label)'] = data['Sentiment (Label)'].apply(Converting_To_Primitive)
    data['Text'] = data['Text'].apply(lemmatization_text)
    return data

def text_preprocess(text): 
    # preprocess Text
    text = text.lower()
    text = remove_stop_words(text)
    text = remove_punctuation(text)
    text = remove_slangs(text)
    text = lemmatization_text(text)
    return text

import sys
import pandas as pd
from pathlib import Path
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtCore import QObject, Slot
from Naive_Bayes_Model import train_naive_bayes_model, predict_sentiment_naive_bayes
from SVM_Model import train_SVM_model, predict_sentiment_SVM
from LogisticRegression_Model import train_LogisticRegression_model, predict_sentiment_LogisticRegression



class MainWindow(QObject):
    def __init__(self, parent: QObject | None = ...) -> None:
        super().__init__()

    @Slot(int, str, result=str)
    def predict(self, active, text):
        if (active == 1):
            print(text)
            return predict_sentiment_naive_bayes(text_preprocess(text))
        elif (active == 2):
            return predict_sentiment_LogisticRegression(text_preprocess(text))
        elif (active == 3):
            return predict_sentiment_SVM(text_preprocess(text))
        else:
            return "wut"


if __name__ == "__main__":
    
    data = pd.read_csv('../sentimentdataset.csv')
    processed_data = pre_process(data)

    train_naive_bayes_model(processed_data)
    train_LogisticRegression_model(processed_data)
    train_SVM_model(processed_data)



    ## gui
    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()
    main = MainWindow()
    engine.rootContext().setContextProperty("test_back_end", main)
    qml_file = Path(__file__).resolve().parent / "main.qml"
    engine.load(qml_file)
    if not engine.rootObjects():
        sys.exit(-1)
    sys.exit(app.exec())
