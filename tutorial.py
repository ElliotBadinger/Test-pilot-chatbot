# If latest version of python is giving you problems, use python 3.6.4
from random import random

import random
import json
import pickle
import numpy as np

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
# (under this line of code) run nltk.download('punkt') if you get an error message saying 'punkt' is not installed
from nltk.stem import WordNetLemmatizer

from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Dropout
from keras.optimizers import gradient_descent_v2
 
lemmatizer = WordNetLemmatizer
wnl = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '`', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [wnl.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

print(words)

# Continue watching chatbot video located on your flashdrive in the folder /Coding and start from time 14:30
