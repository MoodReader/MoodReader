import string

def remove_punctuation(text):
    additional_punctuation = '💪🐾🎉🎨🎂❤️🐶'
    all_punctuation = string.punctuation + additional_punctuation
    return ''.join([c for c in text if c not in all_punctuation])