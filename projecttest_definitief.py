import os
import xml.etree.ElementTree as ET
import nltk
import re
import xmltodict
from bs4 import BeautifulSoup
import json
import random
from random import shuffle, seed
import itertools
import numpy as np
import matplotlib.pyplot as plt


#make subset of English training data
def subset(path): 
	teens = []
	tweens = []
	thirties = []
	categorie1 = re.compile(r'10s_(male|female)\.xml$')
	categorie2 = re.compile(r'20s_(male|female)\.xml$')
	categorie3 = re.compile(r'30s_(male|female)\.xml$')
	for filename in os.listdir(path):
		if categorie1.search(filename):
			teens.append(os.path.join(path, filename))
		if categorie2.search(filename):
			tweens.append(os.path.join(path, filename))
		if categorie3.search(filename):
			thirties.append(os.path.join(path, filename))

	categories = [teens, tweens, thirties]
	random.seed(0.47231099848)
	subset = [fname for flist in categories for fname in random.sample(flist, 5000)]
	#print(len(subset)) #balanced subset of 5000 10s, 5000 20s, 5000 30s
	return subset

subset_list = subset('\\Users\Lorien\Documents\Amaster\Comptaalbegrip\project\pan13-author-profiling-training-corpus-2013-01-09\en')
 #print(subset_list) 

#### Preprocessing #### : make a list of the valuable info: age and text + Remove html mark-up, urls, unnecessary characters. No stop word removal. 

clean_data = []
for file in subset_list:
	tree = ET.parse(file)
	#print(tree)
	root = tree.getroot()
	#print(root)
	textandlabel = []
	for conv in root.findall('./conversations/conversation'):
		#print(conv.text)
		conversation = []
		soup = BeautifulSoup(conv.text, "lxml")
		soup = soup.get_text() #remove HTML markup
		#print(soup)
		pattern1 = re.compile(r'(url|.www\s?\.\s?|(http[s]?:\s?//))(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+') #remove URLs
		clean_text1 = pattern1.sub(' ', soup)
		pattern2 = re.compile(r';') #remove remaining unwanted characters
		clean_text2 = pattern2.sub(' ', clean_text1)
		pattern3 = re.compile(r'\s') #to remove \n and \t
		clean_text = pattern3.sub(' ', clean_text2)
		conversation.append(clean_text)
		age = root.attrib['age_group']
		conversation.append(age)
		textandlabel.append(conversation)
	clean_data.append(textandlabel) # nested list with 15000 lists, 1 list per xml file
	#print(len(clean_data)) 
	#print(clean_data)


#write to Json file and shuffle data 
with open('clean_data.json','w')as fp: #dump list into jsonfile
	json.dump(clean_data, fp)

def seed():
	return 0.47231099848
with open('clean_data.json', 'r') as f:
	data = json.load(f)
random.shuffle(data,seed)
#print("Number of instances:", len(data))

#### feature engineering ####

instances = []
labels = []

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import ngrams, skipgrams, FreqDist, pos_tag
for list1 in data:
	for text, age in list1: 
		tokens = word_tokenize(text)# Tokenize the text
		if len(tokens) < 5: #If conversation is shorter than five tokens, ignore it.
			continue
		wnl = WordNetLemmatizer()
		lemmas = [wnl.lemmatize(t) for t in tokens]
		sents = sent_tokenize(text)
				# POS-tag the text
		tagged = pos_tag(tokens) 
		

		vector = {} #Initiate vector for each conversation

# Features are grouped in blocks, so it is easier to switch between and test the different groups
#### Numeric/word-based features ####
		num_words = 0 #Type-token ratio
		words = {}
		for token in tokens:
			num_words += 1
			if token in words:
				words[token] += 1
			else:
				words[token] = 1
		TTR = str(len(words) / num_words)
		vector ['TTR'] = TTR

		tnw = len(tokens) # total number of words
		vector['tnw'] = tnw			


		awl = sum([len(token) for token in tokens]) / len(tokens)# Average word length
		asl = len(tokens) / len(sents) # Average sentence length
		vector['awl'] = awl
		vector['asl'] = asl
			

#### relative frequency of different Ngrams ####: 

		for n in [1]: #unigrams 
			grams = ngrams(tokens, n)
			fdist = FreqDist(grams)
			total = sum(c for g,c in fdist.items())
		for gram, count in fdist.items():
			vector['w'+str(n)+'+'+' '.join(gram)] = count/total


		for n in [2]: #bigrams
			grams = ngrams(tokens, n)
			fdist = FreqDist(grams)
			total = sum(c for g,c in fdist.items())
		for gram, count in fdist.items():
			vector['w'+str(n)+'+'+' '.join(gram)] = count/total  
		

##### POS ngram relative frequency #### 
		tags = list(zip(*tagged))[1]
		for n in [1]: #unigrams 
			grams = ngrams(tags, n)
			fdist = FreqDist(grams)
			total = sum(c for g,c in fdist.items())
		for gram, count in fdist.items():
			vector['p'+str(n)+'+'+' '.join(gram)] = count/total

		for n in [2]: #bigrams
			grams = ngrams(tags, n)
			fdist = FreqDist(grams)
			total = sum(c for g,c in fdist.items())
		for gram, count in fdist.items():
			vector['p'+str(n)+'+'+' '.join(gram)] = count/total

		labels.append(age.strip().lower())
		instances.append(vector)
#print(len(labels))
#print(len(instances))
#print(vector)



#### Scikits feature extraction module ####
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction import DictVectorizer

vec = DictVectorizer()
X = vec.fit_transform(instances) # No split between train and test data --> later use of 10-fold cross validation
scaler = StandardScaler(with_mean=False) 
X_scaled = scaler.fit_transform(X) # same scale for everything

enc = LabelEncoder()
y = enc.fit_transform(labels)

#### feature selection ####
from sklearn.feature_selection import SelectKBest, chi2

feat_sel = SelectKBest(chi2, k = 10000) #select the 10000 best features
X_fs = feat_sel.fit_transform(X_scaled, y)
#print(X_fs.shape)


#### 10 fold cross validation to train and evaluate the classifier ####
from sklearn import model_selection
from sklearn.metrics import classification_report

#Experiment with decision tree classifier
#from sklearn.tree import DecisionTreeClassifier
#clf = DecisionTreeClassifier()

#Experiment with random forest classifier
#from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier()

#Experiment with linear SVC
from sklearn.svm import LinearSVC
clf = LinearSVC()

y_pred = model_selection.cross_val_predict(clf, X_fs, y, cv=10)


#### Evaluation metrics ####
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report

print(classification_report(y, y_pred, target_names=enc.classes_))

p,r,f,s = precision_recall_fscore_support(y, y_pred, pos_label=None, average='macro')

#print('Precision_M', p) 
#print('Recall_M', r) 
print('Fscore_M',f)#0.63

#This function prints and plots the confusion matrix.
def plot_confusion_matrix(cm, classes,
						  normalize = False,
						  title = 'Confusion Matrix',
						  cmap = plt.cm.Blues):
#diagonal elements represent number of conversations for which predicted label is equal to true label,
#while off-diagonal elements are those that are mislabeled. The higher the diagonal values, the better the predictions
    plt.imshow(cm, interpolation = 'nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)
    if normalize:
    	cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    	print('Normalized confusion matrix')
    else:
    	print('Confusion Matrix without normalization')
    print(cm)
    thresh = cm.max()/2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    	plt.text(j, i, cm[i, j],
    	         horizontalalignment = 'center',
                 color = 'white' if cm[i, j]> thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

target_names = enc.classes_

#compute matrix
cnf_matrix  = confusion_matrix(y, y_pred)
np.set_printoptions(precision = 2)

#plot non-normalized matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes = target_names,
 					  title = 'Confusion matrix without normalization')

#plot normalized matrix, this normalization gives a more visual interpretation of which class is being misclassified 
plt.figure()
plot_confusion_matrix(cnf_matrix, classes = target_names, normalize = True,
                      title = 'Normalized confusion matrix')
plt.show()


#### Calculating the baseline ####: for comparison of the results, a baseline result can tell you whether a change is adding value

def wrb(distribution):
	"""
	Calculate weighted random baseline of a class distribution.
	The variable 'distribution' is a list containing the relative frequency 
	(proportion, thus float between 0 and 1) of each class.  
	"""
	sum = 0
	for prop in distribution:
		sum += prop**2
	return sum
one = labels.count('10s')/len(labels)
#print(one) 
two = labels.count('20s')/len(labels) 
#print(two)
three = labels.count('30s')/len(labels) 
#print(three)
distr = [one,two,three]
print('Majority baseline', max(distr)) #0.35
print('WRB', wrb(distr))#0.33