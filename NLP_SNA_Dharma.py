#Import the following libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import re
from textblob import TextBlob

#Import the dataset
df= pd.read_csv('Dune_movie_review.csv')
#make an ID for the dataset
import uuid
id = uuid.uuid1()
print(id.int)
#Data preprocessing
columns = ['reviewerID','reviewerName','reviewText','sentiment']
review_properties = []
for index,review in df.tail(300).iterrows() :
  properties =[]
  id = uuid.uuid1()
  properties.append(id)
  properties.append(review['audience-reviews__name'])
  review_bersih = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",review['audience-reviews__review']).split())
  properties.append(review_bersih)
  review_properties.append(properties)
  analisis = TextBlob(review_bersih)
  if analisis.sentiment.polarity >0.0:
    properties.append("positif")
  elif analisis.sentiment.polarity ==0.0:
    properties.append("netral")
  else:
    properties.append("negatif")
    # print(tweet_properties)
dr = pd.DataFrame(data=review_properties,columns = columns)

#print the result 
bar = dr.groupby('sentiment').count()['reviewerID'].sort_values(ascending=False)
print(bar)
plt.figure(figsize=(3, 1))
plt.bar(bar.index, bar)
plt.xticks(rotation=55)
plt.xlabel('sentiment')
plt.ylabel('reviwerID')
plt.show()
                 