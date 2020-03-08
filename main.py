#!/usr/bin/env python

# Useful documentation:
# https://cloud.google.com/blog/products/ai-machine-learning/how-to-serve-deep-learning-models-using-tensorflow-2-0-with-cloud-functions

import numpy as np
import pandas as pd
import pickle
import json
from scipy import stats
from google.cloud import storage
import os
import logging
import re

# ########## 1. PREPARE ########## #

# Define models in the global context to support warm starts
embeddings = None
model_initial = None
model_final = None


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


def download_blob(bucket_name, source_blob_name, target_file_name):
    """
    Download model artifacts stored as blobs in GCP Storage
    """

    try:
        # Connect to Storage
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)

        # Instantiate model Storage blobs
        blob = bucket.blob(source_blob_name)

        # Download model blobs
        print(f'Downloading blob: {source_blob_name}')
        blob.download_to_filename(target_file_name)
        print(f'Blob {source_blob_name} successfully'
              f' downloaded to {target_file_name}')
    except Exception as err:
        logging.error(err)
        raise err


def load_models():
    """
    Download and load embeddings and models if they don't already exist in the
    temporary file system (i.e. the cloud function is being run coldly).
    """

    global embeddings
    global model_initial
    global model_final

    if embeddings is None:
        download_blob(
            bucket_name=os.environ['BUCKET_NAME'],
            source_blob_name=os.environ['EMBEDDINGS_BLOB'],
            target_file_name=os.environ['EMBEDDINGS_FILENAME']
        )
        embeddings = load_embeddings(os.environ['EMBEDDINGS_FILENAME'])

    if model_initial is None:
        download_blob(
            bucket_name=os.environ['BUCKET_NAME'],
            source_blob_name=os.environ['MODEL_INITIAL_BLOB'],
            target_file_name=os.environ['MODEL_INITIAL_FILENAME']
        )
        model_initial = pickle.load(
            open(os.environ['MODEL_INITIAL_FILENAME'], 'rb')
        )

    if model_final is None:
        download_blob(
            bucket_name=os.environ['BUCKET_NAME'],
            source_blob_name=os.environ['MODEL_FINAL_BLOB'],
            target_file_name=os.environ['MODEL_FINAL_FILENAME']
        )
        model_final = pickle.load(
            open(os.environ['MODEL_FINAL_FILENAME'], 'rb')
        )

# ########## 2. EXECUTE ########## #


def words_to_sentiment(words):
    global embeddings

    vecs = embeddings.loc[words].dropna()
    log_odds = vecs_to_sentiment(vecs)

    return pd.DataFrame({'sentiment': log_odds}, index=vecs.index)


def vecs_to_sentiment(vecs):
    global model_initial

    # predict_log_proba gives the log probability for each class
    predictions = model_initial.predict_log_proba(vecs)

    # To see an overall positive vs. negative classification in one number,
    # we take the log probability of positive sentiment minus the log
    # probability of negative sentiment.
    return predictions[:, 1] - predictions[:, 0]


def text_to_sentiment(text):
    # The regex below finds tokens that start with a word-like character (\w),
    # and continues matching characters (.+?) until the next word break (\b).
    # It's a relatively simple expression that manages to extract something very
    # much like words from text.
    token_re = re.compile(r"\w.*?\b")

    tokens = [token.casefold() for token in token_re.findall(text)]
    sentiments = words_to_sentiment(tokens)
    return sentiments['sentiment'].to_frame()


def tweet_positivity_scorer(tweet):
    global embeddings
    global model_final

    text = text_to_sentiment(tweet)

    text["Positivity_Score_y"] = text["sentiment"].mean()

    try:
        text["Positivity_Score_SD"] = text["sentiment"].std()
    except:
        text["Positivity_Score_SD"] = 0

    text.loc[text['sentiment'] < 0, 'Positivity_Score_z_temp'] = 'Negative Number'
    text.loc[text['sentiment'] > 0, 'Positivity_Score_z_temp'] = 'Positive Number'

    text["Positive_Score_z"] = text.groupby('Positivity_Score_y')["Positivity_Score_z_temp"].transform(
        lambda x: x.mode().iloc[0])
    text["Positivity_score_x"] = text['sentiment']

    del text['sentiment']
    del text["Positivity_Score_z_temp"]

    text["Word"] = text.index

    embeddings["Word"] = embeddings.index

    text = pd.merge(text, embeddings, on="Word", how="left")

    del text["Word"]
    del embeddings["Word"]

    text = pd.get_dummies(text, columns=['Positive_Score_z'])

    for i in ("Positive_Score_z_Positive Number", "Positive_Score_z_Negative Number",
              "Positive_Score_z_['Negative Number' 'Positive Number']"):
        if i not in text.columns:
            text[i] = 0

    try:
        output = ((stats.mode(model_final.predict(text)))[0][0])
    except:
        output = 2

    if output == 4 and text["Positivity_Score_y"][0] > 0.5 and text["Positive_Score_z_Positive Number"][0] == 1:
        return (tweet, abs(text["Positivity_Score_y"][0]), "Positive Tweet")
    else:
        return (tweet, -abs(text["Positivity_Score_y"][0]), "Bad Tweet")


def moody(request):
    """
    HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
            <http://flask.pocoo.org/docs/1.0/api/#flask.Request>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>.
    """

    try:
        print('Loading models...')
        load_models()
        print('Models loaded successfully!')

        print('Calling tweet_positivity_scorer')
        score = tweet_positivity_scorer("We will get this to work")
        print(f'Alvin model successfully ran! Result = {score}')

        return score
    except Exception as err:
        logging.error('Alvin model fucked up!')
        logging.error(err)

        return 'Fucked up!'

    # json_dict = dict()
    # json_dict['tweet'] = score[0]
    # json_dict['positivity_score'] = score[1]
    # json_dict['positivity_rating'] = score[2]
    # print(json_dict)
    #
    # app_json = json.dumps(json_dict)
    # print(app_json)
