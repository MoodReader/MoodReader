import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()

def lemmatization_text(sentence):
    
	def pos_tagger(nltk_tag):
		if nltk_tag.startswith('J'):
			return wordnet.ADJ
		elif nltk_tag.startswith('V'):
			return wordnet.VERB
		elif nltk_tag.startswith('N'):
			return wordnet.NOUN
		elif nltk_tag.startswith('R'):
			return wordnet.ADV
		else:
			return None

	# tokenize the sentence and find the POS tag for each token
	pos_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
	# we use our own pos_tagger function to make things simpler to understand.
	wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))

	lemmatized_sentence = []
	for word, tag in wordnet_tagged:
		if tag is None:
			# if there is no available tag, append the token as is
			lemmatized_sentence.append(word)
		else:
			# else use the tag to lemmatize the token
			lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
	lemmatized_sentence = " ".join(lemmatized_sentence)

	return lemmatized_sentence