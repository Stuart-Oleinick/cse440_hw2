
import os
import nltk
import pandas
import collections
import itertools
import random
import numpy as np


K = 0.1
POS_PATH = './posTrain/'
NEG_PATH = './negTrain/'
POS_TEST_PATH = './posTest/'
NEG_TEST_PATH = './negTest/'

tokenizer = nltk.tokenize.RegexpTokenizer(r'[a-zA-Z]+[\']*[a-zA-Z]+|[;!?$]')

#gets all the testing files
posTestList = [POS_TEST_PATH+f for f in os.listdir(POS_TEST_PATH)]
negTestList = [NEG_TEST_PATH+f for f in os.listdir(NEG_TEST_PATH)]

testing_documents = []

for file in posTestList:
    with open(file,'r') as f:
        rawText = f.read()
        cleanText = tokenizer.tokenize(rawText)
        testing_documents.append((cleanText,'pos'))

for file in negTestList:
    with open(file,'r') as f:
        rawText = f.read()
        cleanText = tokenizer.tokenize(rawText)
        testing_documents.append((cleanText,'neg'))

# Get all the training files.. this will take a sec
posFileList = [POS_PATH+f for f in os.listdir(POS_PATH)]
negFileList = [NEG_PATH+f for f in os.listdir(NEG_PATH)]

training_documents = []

for file in posFileList:
    with open(file,'r') as f:
        rawText = f.read()
        cleanText = tokenizer.tokenize(rawText)
        training_documents.append((cleanText,'pos'))

for file in negFileList:
    with open(file,'r') as f:
        rawText = f.read()
        cleanText = tokenizer.tokenize(rawText)
        training_documents.append((cleanText,'neg'))

random.shuffle(training_documents)
all_words = []

for tup in training_documents:
    words_list = tup[0]
    for word in words_list:
        all_words.append(word.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

training_featuresets = [(find_features(rev),category) for (rev,category) in training_documents]
testing_featuresets =  [(find_features(rev),category) for (rev,category) in testing_documents]


classifier = nltk.NaiveBayesClassifier.train(training_featuresets)

print("Naive Bayes accuracy percent:",(nltk.classify.accuracy(classifier,testing_featuresets))*100)

classifier.show_most_informative_features(15)
