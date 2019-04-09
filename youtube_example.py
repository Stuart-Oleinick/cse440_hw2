import os
import pandas
import collections
import itertools
import numpy as np
import nltk
import random
from nltk.corpus import movie_reviews
#nltk.download('movie_reviews')
tokenizer = nltk.tokenize.RegexpTokenizer(r'[a-zA-Z]+[\']*[a-zA-Z]+|[;!?$]')

documents = [(list(movie_reviews.words(fileid)),category) \
for category in movie_reviews.categories() \
for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

#print(documents[0])
all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

#print(find_features('neg/cv000_29416.txt'))

featuresets = [(find_features(rev),category) for (rev,category) in documents]

#print(featuresets[0])

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

print(type(training_set))

classifier = nltk.NaiveBayesClassifier.train(training_set)

print("Naive Bayes accuracy percent:",(nltk.classify.accuracy(classifier,testing_set))*100)

classifier.show_most_informative_features(15)
