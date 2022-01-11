import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 500)

# Gather the independent variables in a dataframe and the labels in another
df1 = pd.read_csv("True.csv")
df2 = pd.read_csv("Fake.csv")

df = pd.concat([df1, df2], ignore_index=True)

X = list(df["text"])

# Lav target array, hvor 0 er en troværdig artikel og 1 er en utroværdig artikel
y1 = np.zeros(len(df1))
y2 = np.ones(len(df2))
y = np.hstack((y1, y2))
print(y)

print(X[1])
# Remove all non-alphabetical characters such as /, !, ., etc...
# Stem all the words in the "text" part of x and add them to a list.
# Afterwards use the class "CountVectorizer" to gather all documents in a matrix as bags of words.
# Remove the most common words from the array

import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
corpus = []

# Iterer over artiklerne og fjern specialtegn
for text in X:
    text_oa = re.sub("[^a-zA-Z]", " ", text)

    text_oa = text_oa.lower()
    text_oa = text_oa.split()

    # stopwords filtreres fra, og ordene stemmes

    filtered_article = [ps.stem(w) for w in text_oa if w not in stop_words and len(w) > 1]
    corpus.append(" ".join(filtered_article))

print(corpus[0])

# Corpus omdannes herefter til en bag_of_words repræsentation
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(corpus)

print("Vocabulary size: {}".format(len(vect.vocabulary_)))

bag_of_words = vect.transform(corpus)
print("bag_of_words: {}".format(repr(bag_of_words)))

# Dataen opdeles nu i en træningsdel og en testdel
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(bag_of_words, y, random_state=0)
print("Training articles: {}\nTest articles: {}".format(np.size(y_train), np.size(y_test)))

# Modellen trænes med træningsdataen
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

print("Test set accuracy: {}".format(mnb.score(X_test, y_test)))

