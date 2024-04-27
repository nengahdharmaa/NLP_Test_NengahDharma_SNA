## Import Library 
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import uuid
# Natural Language ToolKit Library
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# Model classification 
from sklearn.naive_bayes import MultinomialNB
# Saving the model
import pickle

nltk.download('stopwords')
df = pd.read_csv('labeled_Movie_review_sentiment_dune.csv')

## Stemming the data
stem_df = PorterStemmer()
def stemming_df(review):
  review_bersih = (re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",review).lower()).split()
  review_bersih = [stem_df.stem(word) for word in review_bersih if not word in stopwords.words('english')]
  review_bersih = ' '.join(review_bersih)
  return review_bersih

df['review'] = df['audience-reviews__review'].apply(stemming_df)

## Splitting Dataset 
X = df['review'].values
y = df['Sentiment'].values
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.2,random_state=2)

## Convert the text review to numerical values
#Converting into numerical data
vectorizer = TfidfVectorizer()
Xtrain = vectorizer.fit_transform(Xtrain)
Xtest = vectorizer.transform(Xtest)

## Import the Naive Bayes model 
mnb = MultinomialNB()
modelnb = mnb.fit(Xtrain,ytrain)
ypred1 = modelnb.predict(Xtest)

## Evaluate the model 
#Accuracy Score on The training data
print('Accuracy Score on the training data :', accuracy_score(ytrain,modelnb.predict(Xtrain)))
#Accuracy Score on the test data
print('Accuracy Score on the test data :',accuracy_score(ytest,modelnb.predict(Xtest)))