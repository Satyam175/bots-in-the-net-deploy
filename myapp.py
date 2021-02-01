# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 20:31:01 2021

@author: user
"""


import streamlit as st
import pandas as pd

import pickle
from keras.models import model_from_yaml
import re
from nltk.tokenize import TweetTokenizer
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
additional  = ['rt','rts','retweet']
stop_words = set().union(stopwords.words('english'),additional)
lemmatizer = WordNetLemmatizer()
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot

#%% Deployment

st.write("""
# Who Are YOU???
""")

#st.sidebar.header('User Input Parameters')

def user_input_features():
    message = st.text_area("Tweet here...")
    if st.button("Submit"):
          result = message.title()
          #st.success(result)
    
	   
    
    return message

df = user_input_features()

#st.subheader('User Input parameters')
#st.write(df)

# %% Loading the model
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
#load YAML and create model
yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)

# load weights into new model
loaded_model.load_weights("model.h5")

 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

spec=[]
def clean_word(text):
    tweet = re.sub('[^a-zA-Z]', ' ', text)
    
    tweet = tweet.lower()
    tweet = tweet.split()
    tweet = [ps.stem(word) for word in tweet if not word in stopwords.words('english')]
    tweet = ' '.join(tweet)
    spec.append(tweet)


clean_word(df)
vocab_size=5000
one_hot_re=[one_hot(word,vocab_size) for word in spec]
sent_length=40
embedded=pad_sequences(one_hot_re,padding='pre',maxlen=sent_length)

pred=loaded_model.predict_classes(embedded)


if pred==0:
    pred='You are Human'
else:
    pred='You are BOT'

#print(prediction)
st.subheader("Prediction")
st.write(pred)

















# clf=pickle.load(open('credit_loan_model','rb'))

# prediction = clf.predict(df)
# prediction_proba = clf.predict_proba(df)
# if prediction==0:
#     prediction='Non-Defaulter'
# else:
#     prediction='Defaulter'
# st.subheader("Prediction")
# st.write(prediction)

# st.subheader('Prediction')
# st.write(iris.target_names[prediction])
# #st.write(prediction)

# st.subheader('Prediction Probability')
# st.write(prediction_proba)