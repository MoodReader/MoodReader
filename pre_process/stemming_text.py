from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()


def stemming_text(text):
    words = word_tokenize(text)
    stemmed_string = ' '.join([ps.stem(word) for word in words])
    return stemmed_string
