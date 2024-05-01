import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer


# func: .apply takes a function and applies it to all values of pandas series. 
# convert_dtype: Convert dtype as per the function’s operation.
# args=(): Additional arguments to pass to function instead of series.
# Return Type: Pandas Series after applied function/operation.

def remove_stop_words(text):
    # make the whole sentence into small peices
    tokenizer = TweetTokenizer()
    sentence = tokenizer.tokenize(text)
    # stop words for lang -> english
    stop_words = set(stopwords.words('english'))
    sentence_without_stopwords = []
    for word in sentence:
        if word not in stop_words:
            sentence_without_stopwords.append(word)
    processed_text = ' '.join(sentence_without_stopwords)
    # The line processed_text = ' '.join(tokens_without_stopwords) joins the
    # list tokens_without_stopwords into a single string, separated by a space ' '.
    return processed_text