
import os
import nltk
import pandas
import collections
import itertools
import random
import numpy as np
import string


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
        #cleanText = [word for word in cleanText if word not in string.punctuation]
        testing_documents.append((cleanText,'pos'))

for file in negTestList:
    with open(file,'r') as f:
        rawText = f.read()
        cleanText = tokenizer.tokenize(rawText)
        #cleanText = [word for word in cleanText if word not in string.punctuation]
        testing_documents.append((cleanText,'neg'))

random.shuffle(testing_documents)

# Get all the training files.. this will take a sec
posFileList = [POS_PATH+f for f in os.listdir(POS_PATH)]
negFileList = [NEG_PATH+f for f in os.listdir(NEG_PATH)]

training_documents = []

for file in posFileList:
    with open(file,'r') as f:
        rawText = f.read()
        cleanText = tokenizer.tokenize(rawText)
        #cleanText = [word for word in cleanText if word not in string.punctuation]
        training_documents.append((cleanText,'pos'))

for file in negFileList:
    with open(file,'r') as f:
        rawText = f.read()
        cleanText = tokenizer.tokenize(rawText)
        #cleanText = [word for word in cleanText if word not in string.punctuation]
        training_documents.append((cleanText,'neg'))

random.shuffle(training_documents)
all_words = []

for tup in training_documents:
    words_list = tup[0]
    for word in words_list:
        all_words.append(word.lower())

all_words = nltk.FreqDist(all_words)

#print(len(all_words))

word_features = list(all_words.keys())

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

training_featuresets = [(find_features(rev),category) for (rev,category) in training_documents]
testing_featuresets =  [(find_features(rev),category) for (rev,category) in testing_documents]


#classifier = nltk.NaiveBayesClassifier.train(training_featuresets)

#print("Naive Bayes accuracy percent:",(nltk.classify.accuracy(classifier,testing_featuresets))*100)

#classifier.show_most_informative_features(15)

#print(training_featuresets[0])

right_wrong = 0

training_dict = {}
pos_words = []
neg_words = []


# for tup in training_documents:
#     words = tup[0]
#     pos_neg = tup[1]
#     for w in words:
#         if w not in training_dict:
#             training_dict[w] = 0
#         if pos_neg == 'pos':
#             training_dict[w]+=1
#         elif pos_neg == 'neg':
#             training_dict[w]-=1


for tup in training_featuresets:
    feature_dict = tup[0]
    pos_neg = tup[1]
    for k,v in feature_dict.items():
        if k not in training_dict:
            training_dict[k] = 0
        if v:
            if pos_neg == 'pos':
                training_dict[k]+=1
            elif pos_neg == 'neg':
                training_dict[k]-=1

#print(training_dict)

for k,v in training_dict.items():
    if v>0:
        pos_words.append(k)
    elif v<0:
        neg_words.append(k)


for tup in testing_documents:
    count = 0
    words = tup[0]
    pos_neg = tup[1]
    for w in words:
        if w in training_dict:
            # if training_dict[w]>100 or training_dict[w]<-100:
                # print(w)
                # print(training_dict[w])
                # print()
                count+=training_dict[w]
    if count>0 and pos_neg == 'pos':
        right_wrong+=1
    elif count<0 and pos_neg == 'neg':
        right_wrong+=1

    #print('\ncount:',count)
    #print('pos neg:',pos_neg)
    #print("--------------------------------------------------")

total = len(testing_documents)

print('accuracy:',(right_wrong/total)*100)
