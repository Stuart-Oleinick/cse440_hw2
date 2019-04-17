
import os
import nltk
import pandas
import collections
import itertools
import random
import numpy as np
import string
import math


K = 0.1
POS_PATH = './posTrain/'
NEG_PATH = './negTrain/'
POS_TEST_PATH = './posTest/'
NEG_TEST_PATH = './negTest/'

tokenizer = nltk.tokenize.RegexpTokenizer(r'[a-zA-Z]+[\']*[a-zA-Z]+|[;!?$]')

posFileList = [POS_PATH+f for f in os.listdir(POS_PATH)]
negFileList = [NEG_PATH+f for f in os.listdir(NEG_PATH)]


pos_words = {}
total_pos_words = 0

#gathering the count of each word in each pos and neg reviews
for file in posFileList:
    with open(file,'r',encoding='ISO-8859-1') as f:
        rawText = f.read()
        cleanText = tokenizer.tokenize(rawText)
        for word in cleanText:
        	if word not in pos_words:
        		pos_words[word] = 0
        	pos_words[word]+=1
        	total_pos_words+=1

neg_words = {}
total_neg_words = 0

for file in negFileList:
    with open(file,'r',encoding='ISO-8859-1') as f:
        rawText = f.read()
        cleanText = tokenizer.tokenize(rawText)
        for word in cleanText:
        	if word not in neg_words:
        		neg_words[word] = 0
        	neg_words[word]+=1
        	total_neg_words+=1

#calculating the probability of a word in pos and neg  using smoothing

pos_prob = {}
for k,v in pos_words.items():
	#(xi+0.1)/N+0.1*d
	pos_prob[k] = (v+.1)/(total_pos_words+0.1*len(pos_words.keys())) 

neg_prob = {}
for k,v in neg_words.items():
	#(xi+0.1)/N+0.1*d
	neg_prob[k] = (v+.1)/(total_neg_words+0.1*len(neg_words.keys())) 

#adding unique words in pos to neg and vise versa
for k,v in pos_words.items():
	if k not in neg_words:
		neg_prob[k] = (.1)/(total_neg_words+0.1*len(neg_words.keys())) 

for k,v in neg_words.items():
	if k not in pos_words:
		pos_prob[k] = (.1)/(total_pos_words+0.1*len(pos_words.keys())) 


#gets all the testing files
posTestList = [POS_TEST_PATH+f for f in os.listdir(POS_TEST_PATH)]
negTestList = [NEG_TEST_PATH+f for f in os.listdir(NEG_TEST_PATH)]

testing_documents = []


total_right = 0
for file in posTestList:
	pos_temp = 0
	neg_temp = 0
	with open(file,'r',encoding='ISO-8859-1') as f:
		rawText = f.read()
		cleanText = tokenizer.tokenize(rawText)

	for w in cleanText: #if the word is in a pos or neg review, add its log(probability)
		if w in pos_prob:
			pos_temp+=math.log(pos_prob[w])
		if w in neg_prob:
			neg_temp+=math.log(neg_prob[w])
	if (pos_temp>neg_temp):
		total_right+=1

for file in negTestList:
	pos_temp = 0
	neg_temp = 0
	with open(file,'r',encoding='ISO-8859-1') as f:
		rawText = f.read()
		cleanText = tokenizer.tokenize(rawText)

	for w in cleanText: #if the word is in a pos or neg review, add its log(probability)
		if w in pos_prob:
			pos_temp+=math.log(pos_prob[w])
		if w in neg_prob:
			neg_temp+=math.log(neg_prob[w])
	if(pos_temp<neg_temp):
		total_right+=1

print('accuracy:', "{:.3%}".format(total_right/(len(posTestList)+len(negTestList))))





