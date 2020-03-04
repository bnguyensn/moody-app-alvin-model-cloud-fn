#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib
import seaborn
from sklearn.metrics import f1_score
import re
import statsmodels.formula.api

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
import json
from scipy import stats


# In[3]:


def load_embeddings(filename):
    """
    Load a DataFrame from the generalized text format used by word2vec, GloVe,
    fastText, and ConceptNet Numberbatch. The main point where they differ is
    whether there is an initial line with the dimensions of the matrix.
    """
    labels = []
    rows = []
    with open(filename, encoding='utf-8') as infile:
        for i, line in enumerate(infile):
            items = line.rstrip().split(' ')
            if len(items) == 2:
                # This is a header row giving the shape of the matrix
                continue
            labels.append(items[0])
            values = np.array([float(x) for x in items[1:]], 'f')
            rows.append(values)

    arr = np.vstack(rows)
    return pd.DataFrame(arr, index=labels, dtype='f')


# In[4]:


embeddings = load_embeddings('C:/Users/944382/Desktop/Sentiment Analysis Playaround/glove.42B.300d.txt')
embeddings.shape

# In[5]:


model_initial = pickle.load(open(r'C:\Users\944382\Desktop\Sentiment Analysis Playaround\initial_model.sav', 'rb'))

# In[6]:


model_final = pickle.load(open(r'C:\Users\944382\Desktop\Sentiment Analysis Playaround\rf_model.sav', 'rb'))


# In[7]:


def words_to_sentiment(words):
    vecs = embeddings.loc[words].dropna()
    log_odds = vecs_to_sentiment(vecs)
    return pd.DataFrame({'sentiment': log_odds}, index=vecs.index)


def vecs_to_sentiment(vecs):
    # predict_log_proba gives the log probability for each class
    predictions = model_initial.predict_log_proba(vecs)

    # To see an overall positive vs. negative classification in one number,
    # we take the log probability of positive sentiment minus the log
    # probability of negative sentiment.
    return predictions[:, 1] - predictions[:, 0]


# In[8]:


import re

TOKEN_RE = re.compile(r"\w.*?\b")


# The regex above finds tokens that start with a word-like character (\w), and continues
# matching characters (.+?) until the next word break (\b). It's a relatively simple
# expression that manages to extract something very much like words from text.


def text_to_sentiment(text):
    tokens = [token.casefold() for token in TOKEN_RE.findall(text)]
    sentiments = words_to_sentiment(tokens)
    return sentiments['sentiment'].to_frame()


# In[9]:


test = text_to_sentiment("Hi how are you")


# In[10]:


def Tweet_Positivity_scorer(tweet):
    Text = text_to_sentiment(tweet)
    Text["Positivity_Score_y"] = Text["sentiment"].mean()
    try:
        Text["Positivity_Score_SD"] = Text["sentiment"].std()
    except:
        Text["Positivity_Score_SD"] = 0
    Text.loc[Text['sentiment'] < 0, 'Positivity_Score_z_temp'] = 'Negative Number'
    Text.loc[Text['sentiment'] > 0, 'Positivity_Score_z_temp'] = 'Positive Number'
    Text["Positive_Score_z"] = Text.groupby('Positivity_Score_y')["Positivity_Score_z_temp"].transform(
        lambda x: x.mode().iloc[0])
    Text["Positivity_score_x"] = Text['sentiment']
    del Text['sentiment']
    del Text["Positivity_Score_z_temp"]
    Text["Word"] = Text.index
    embeddings["Word"] = embeddings.index
    Text = pd.merge(Text, embeddings, on="Word", how="left")
    del Text["Word"]
    del embeddings["Word"]
    Text = pd.get_dummies(Text, columns=['Positive_Score_z'])
    for i in ("Positive_Score_z_Positive Number", "Positive_Score_z_Negative Number",
              "Positive_Score_z_['Negative Number' 'Positive Number']"):
        if i not in Text.columns:
            Text[i] = 0
    try:
        output = ((stats.mode(model_final.predict(Text)))[0][0])
    except:
        output = 2
    if output == 4 and Text["Positivity_Score_y"][0] > 0.5 and Text["Positive_Score_z_Positive Number"][0] == 1:
        return (tweet, abs(Text["Positivity_Score_y"][0]), "Positive Tweet")
    else:
        return (tweet, -abs(Text["Positivity_Score_y"][0]), "Bad Tweet")


# In[20]:


score = Tweet_Positivity_scorer("We will get this to work")

# In[21]:


json_dict = dict()
json_dict['tweet'] = score[0]
json_dict['positivity_score'] = score[1]
json_dict['positivity_rating'] = score[2]
print(json_dict)

# In[22]:


app_json = json.dumps(json_dict)
print(app_json)

