# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 12:10:06 2021

@author: user
"""

"""

## Importing Libraries
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import keras
import nltk
import unicodedata
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from tensorflow.keras.preprocessing.text import one_hot
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
import json
import random
import csv
import re
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout
import tensorflow as tf
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup
# %matplotlib inline

"""## Dataset Loading"""


f=open('C:/Users/Home/Downloads/tweets-2016-10000-textonly.txt','r', encoding="utf-8")
lines = f.readlines()


tweets = []
labels = []

len_train = 1000

with open('C:/Users/Home/Downloads/IRAhandle_tweets_1.csv', newline='',encoding='utf8') as csvfile:
    categories = csvfile.readline().split(",")
    tweetreader = csv.reader(csvfile, delimiter=',')
    counter = 0
    for row in tweetreader:
        tweet = dict(zip(categories, row))
        if tweet['language'] == 'English':
            tweets.append(tweet['content'])
            labels.append(1)
            counter += 1
        if counter > len_train:
            break
csvfile.close()

# In[ ]:

for line in lines:
    # for line in lines:
    #     tweet = json.loads(line)
    #     if 'user' in tweet.keys():
    #         if tweet["user"]["lang"] == "en":
    #             tweets.append(tweet['text'])
    #             labels.append(0)
    tweets.append(line)
    labels.append(0)

f.close()
            
tweets_to_labels = dict(zip(tweets, labels))
random.shuffle(tweets)

actual = []

for tweet in tweets:
    actual.append(tweets_to_labels[tweet])
data=pd.DataFrame()
data['Text']=tweets
data['labels']=actual
data

"""### Little bit of EDA"""

data['Text'][0]

print ("number of rows for labels 1: {}".format(len(data[data.labels == 1])))
print ("number of rows for labels 0: {}".format(len(data[data.labels == 0])))

data.groupby('labels').describe().transpose()

#Creating a new col
data['length'] = data['Text'].apply(len)
data.head()

"""### Data Visualization"""

#Histogram of count of letters
data['length'].plot.hist(bins = 100)

data.length.describe()

data.hist(column='length', by='labels', bins=50,figsize=(12,4))

"""## Data Cleaning

### Clean the text (remove usernames and links)
"""

def clean(text):
    text = text.fillna("fillna").str.lower()
    text = text.map(lambda x: re.sub('\\n',' ',str(x)))
    text = text.map(lambda x: re.sub("\[\[User.*",'',str(x)))
    text = text.map(lambda x: re.sub("https",'',str(x)))
    text = text.map(lambda x: re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",'',str(x)))
    text = text.map(lambda x: re.sub("\(http://.*?\s\(http://.*\)",'',str(x)))
    return text

data["Text"] = clean(data["Text"])
data['Text'] = data['Text'].str.replace('[^\w\s]','')


"""### More Text Cleaning
applying text cleaning techniques like clean_text,replace_typical_misspell,handle_contractions,fix_quote on train,test and validation set
"""

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
  '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '\xa0', '\t',
  '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '\u3000', '\u202f',
  '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '«',
  '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

mispell_dict = {"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"couldnt" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"doesnt" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"havent" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"shouldnt" : "should not",
"that's" : "that is",
"thats" : "that is",
"there's" : "there is",
"theres" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"theyre":  "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not",
"tryin'":"trying"}


def clean_text(x):
    x = str(x).replace("\n","")
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x


def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x

from nltk.tokenize.treebank import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()

def handle_contractions(x):
    x = tokenizer.tokenize(x)
    return x

def fix_quote(x):
    x = [x_[1:] if x_.startswith("'") else x_ for x_ in x]
    x = ' '.join(x)
    return x

def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re


def replace_typical_misspell(text):
    mispellings, mispellings_re = _get_mispell(mispell_dict)

    def replace(match):
        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)


def clean_data(df, columns: list):
    for col in columns:
#         df[col] = df[col].apply(lambda x: clean_numbers(x))
        df[col] = df[col].apply(lambda x: clean_text(x.lower())) 
        df[col] = df[col].apply(lambda x: replace_typical_misspell(x))
        df[col] = df[col].apply(lambda x: handle_contractions(x))  
        df[col] = df[col].apply(lambda x: fix_quote(x))   
    
    return df




# Commented out IPython magic to ensure Python compatibility.
# %%time
# input_columns = [
#     'Text'   
# ]
# 
# '''applying text cleaning techniques like clean_text,replace_typical_misspell,handle_contractions,fix_quote 
# on train,test and validation set'''
# 
# data = clean_data(data, input_columns)

tokenizer=ToktokTokenizer()
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

X=data.drop('labels',axis=1)
y=data['labels']
y.value_counts()

tweets=X.copy()
tweets.reset_index(inplace=False)
tweets
#%% Data Pre-processing
"""## Data pre-processing"""

from bs4 import BeautifulSoup

#Creating a function for cleaning of data
def clean_text(raw_text):
    
    # 1. remove HTML tags
    raw_text = BeautifulSoup(raw_text).get_text() 
    
    # 2. removing all non letters from text
    letters_only = re.sub("[^a-zA-Z]", "", raw_text) 
    
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower()                          
    
    # 4. Create variable which contain set of stopwords
    stops = set(stopwords.words("english"))                  
    
    # 5. Remove stop word & returning   
    
    
    
    return [w for w in words if not w in stops]

# Commented out IPython magic to ensure Python compatibility.
# %%time
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(tweets)):
    #print(i)
    tweet = re.sub('[^a-zA-Z]', ' ', tweets['Text'][i])
    
    tweet = tweet.lower()
    tweet = tweet.split()
    #tweet = [lemmatizer.lemmatize(token) for token in tweet]
    #tweet = [lemmatizer.lemmatize(token, "v") for token in tweet]
    tweet = [ps.stem(word) for word in tweet if not word in stopwords.words('english')]
    tweet = ' '.join(tweet)
    corpus.append(tweet)



vocab_size=5000
one_hot_repr=[one_hot(words,vocab_size) for words in corpus]
one_hot_repr

sent_length=40
embedded_docs=pad_sequences(one_hot_repr,padding='pre',maxlen=sent_length)
embedded_docs
# %%
"""## Creating LSTM  model"""

# embedding_vector_features=80
# model=keras.Sequential()
# model.add(Embedding(vocab_size,embedding_vector_features,input_length=sent_length))
# model.add(LSTM(100))
# model.add(Dense(1,activation='sigmoid'))
# model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# print(model.summary())

# import numpy as np
# X_final=np.array(embedded_docs)
# y_final=np.array(y)

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)

# #Plot wordcloud
# from wordcloud import WordCloud,STOPWORDS
# word_cloud = WordCloud(width = 1000, height = 500, stopwords = STOPWORDS, background_color = 'white').generate(
#                         ''.join(data['Text']))

# plt.figure(figsize = (15,8))
# plt.imshow(word_cloud)
# plt.axis('off')
# plt.show()

# """## Model Training"""

# # Commented out IPython magic to ensure Python compatibility.
# # %%time
# history=model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)

# y_pred=model.predict_classes(X_test)
# y_pred

# from sklearn.metrics import accuracy_score
# accuracy_score(y_test,y_pred)

# from sklearn.metrics import confusion_matrix, classification_report
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))

# vocab_size=100
# words=['Liz Cheney was given a plum leadership position in the Republican Party. She has not been a good return on the investment -- not in candidate recruitment, not in fundraising, not in supporting the conference, and not in avoiding traps set by media/Pelosi.']
# words=[clean_text(word) for word in words]
# for word in words:
#   corpus.append(word)
# # tokenize the sentences
# tokenizer = Tokenizer(lower=False)
# text_fit=[tokenizer.fit_on_texts(word) for word in corpus]
# text_vec = tokenizer.texts_to_sequences(words)
# # text_vec = pad_sequences(text_vec, maxlen=50)
# # text_vec
# # model.predict_classes(text_vec)
#%% Evaluating the LSTM model
# history.history.keys()
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
#%%
# """### Bidirectional LSTM"""

# embedding_vector_features=80
# model1=keras.Sequential()
# model1.add(Embedding(vocab_size,embedding_vector_features,input_length=sent_length))
# model1.add(keras.layers.Bidirectional(LSTM(200)))
# model1.add(keras.layers.Dropout(0.3))
# model1.add(Dense(1,activation='sigmoid'))
# model1.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# print(model1.summary())

# """#### Finally training BIdirectional LSTM"""
# history1=model1.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)

# y_pred1=model1.predict_classes(X_test)
# y_pred1
# accuracy_score(y_test,y_pred1)
# classification_report(y_test,y_pred1)

# plt.plot(history1.history['accuracy'])
# plt.plot(history1.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()


#%% 
# #%% Predictive Modelling
# import nltk
# nltk.download('wordnet')

tweet1=data.copy()
X1=tweet1['Text'].str.replace('[^\w\s]','')
y1=tweet1['labels']

# Splitting the data in to training and test datasets
from sklearn.model_selection import train_test_split
X_train_1,X_test_1,y_train_1,y_test_1=train_test_split(X1,y1,test_size=0.33,random_state=42)


# Vectorizing the text data
cv=CountVectorizer()
X_train_1=cv.fit_transform(X_train_1)
X_test_1=cv.transform(X_test_1)

# %%Training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
model_nb = MultinomialNB().fit(X_train_1, y_train_1)

from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
y_pred_1=model_nb.predict(X_test_1)
print("Accuracy:",metrics.accuracy_score(y_test_1, y_pred_1))
print('Classification report:',classification_report(y_test_1,y_pred_1))

#%% Training model using Logistic Regression
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train_1,y_train_1)
y_pred_1=lr.predict(X_test_1)
print("Accuracy:",metrics.accuracy_score(y_test_1, y_pred_1))
print('Classification report:',classification_report(y_test_1,y_pred_1))

# %%Training model using Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(random_state=1)
rfc.fit(X_train_1, y_train_1)
y_pred = rfc.predict(X_test_1)
print("Accuracy:",metrics.accuracy_score(y_test_1, y_pred_1))
print('Classification report:',classification_report(y_test_1,y_pred_1))

# %%Training model using SVM
from sklearn import svm
clf = svm.SVC(kernel='rbf', gamma=2)
clf.fit(X_train_1, y_train_1)
y_pred_1 = clf.predict(X_test_1)
print("Accuracy:",metrics.accuracy_score(y_test_1, y_pred_1))
print('Classification report:',classification_report(y_test_1,y_pred_1))

# #%%


# words=['Liz Cheney was given a plum leadership position in the Republican Party. She has not been a good return on the investment -- not in candidate recruitment, not in fundraising, not in supporting the conference, and not in avoiding traps set by media/Pelosi.']
# words=[clean_text(word) for word in words]
# for word in words:
#   corpus.append(word)
# # tokenize the sentences
# tokenizer = Tokenizer(lower=False)
# text_fit=[tokenizer.fit_on_texts(word) for word in corpus]
# text_vec = tokenizer.texts_to_sequences(words)
# text_vec = pad_sequences(text_vec, maxlen=40)

# #%% Saving the model 

# from keras.models import model_from_yaml
# # serialize model to YAML
# model_yaml = model.to_yaml()
# with open("model.yaml", "w") as yaml_file:
#     yaml_file.write(model_yaml)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")
 
# # later...
 
# # load YAML and create model
# yaml_file = open('model.yaml', 'r')
# loaded_model_yaml = yaml_file.read()
# yaml_file.close()
# loaded_model = model_from_yaml(loaded_model_yaml)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")
 
# # evaluate loaded model on test data
# loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# # score = loaded_model.evaluate(X, Y, verbose=0)
# # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

# # %%
# cora=pd.DataFrame(corpus)
# cora.to_csv('cora.csv')
# import pickle
# log=open('LSTM_model.sav','wb')



# from sklearn.pipeline import Pipeline
# pipe=Pipeline(steps=[('cv',CountVectorizer),('model',lr)])


# pickle.dump(pipe,log)
# log.close()
import pickle
log=pickle.load(open('LSTM_model.sav','rb'))

#%%
from keras.models import model_from_yaml
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import text_to_word_sequence

# load YAML and create model
yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# score = loaded_model.evaluate(X, Y, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
 # %%

import string
words='Liz Cheney was given a plum leadership position in the Republican Party. She has not been a good return on the investment not in candidate recruitment, not in fundraising, not in supporting the conference, and not in avoiding traps set by media Pelosi.'

spec=[]
def clean_word(text):
    tweet = re.sub('[^a-zA-Z]', ' ', text)
    
    tweet = tweet.lower()
    tweet = tweet.split()
    #tweet = [lemmatizer.lemmatize(token) for token in tweet]
    #tweet = [lemmatizer.lemmatize(token, "v") for token in tweet]
    tweet = [ps.stem(word) for word in tweet if not word in stopwords.words('english')]
    tweet = ' '.join(tweet)
    spec.append(tweet)



vocab_size=5000
one_hot_re=[one_hot(word,vocab_size) for word in spec]
one_hot_re

sent_length=40
embedded=pad_sequences(one_hot_re,padding='pre',maxlen=sent_length)
embedded


pred=loaded_model.predict_classes(embedded)







