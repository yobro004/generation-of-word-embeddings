import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
#corpus=pd.read_csv(r"E:\dataset cosine\joke.csv")
corpus=[
    'all my cats in a row',
    'when my cats sits down, she looks furry',
    'the cat from outer space',
    'sunshine loves to sit like this for some reason' 
    'my cats name is sunshine']

vectorizer=CountVectorizer()
features=vectorizer.fit_transform(corpus).todense()
#print( vectorizer.vocabulary_ )

for f in features:
    print(euclidean_distances(features[0],f) )
