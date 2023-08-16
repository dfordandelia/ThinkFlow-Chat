 # Roughly identifying what is the type of question/statement asked to the bot by some sort of a classification, using intents.json file for this

import nltk
#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
import numpy as np


stemmer = PorterStemmer()

def tokenize(sentence):

    return nltk.word_tokenize(sentence)

def stem(word):

    return stemmer.stem(word.lower())

def bag_of_words(tokenized_setence, all_words):

    tokenized_setence = [stem(w) for w in tokenized_setence]
    bag = np.zeros(len(all_words),dtype = np.float32)
    for i,word in enumerate(all_words):
        if word in tokenized_setence:
            bag[i] = 1.0
    return bag





