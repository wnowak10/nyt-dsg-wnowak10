#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" data_prep.py

	A module to prepare JSON data for predictive model training.

	Input data of form: 

	{"emotion_0":"0.0","emotion_1":"0.0","emotion_2":"0.0","emotion_3":"0.0","emotion_4":"0.0","emotion_5":"1.0","emotion_6":"0.0","emotion_7":"0.0","emotion_8":"0.0","emotion_9":"1","headline":"Parenting Lessons From a Partial Eclipse","summary":"I don\u2019t do partiality. Maybe it was finally time to try.","worker_id":"95524929"}
	{"emotion_0":"0.0","emotion_1":"0.0","emotion_2":"0.0","emotion_3":"0.0","emotion_4":"0.0","emotion_5":"1.0","emotion_6":"0.0","emotion_7":"0.0","emotion_8":"0.0","emotion_9":"1","headline":"On Catalonia","summary":"Catalan lawmakers declared independence from Spain on Friday.","worker_id":"30266977"}
    
    We clean the data in three key ways:

    1. Cleaning the `emotion_x` labels to ensure we have binary, numeric data 
    with outliers, missing values handled appropriately.

    2. Process text data. In this case, we combine word vectors for each word 
    using a TFIDF weighting.

    3. Perform random train / test split for model assessment.


	Resources:

	- Assignment: https://github.com/chrishwiggins/nyt-dsg-wnowak10
	
"""

# ______________________________________________________________________________
# Imports

import pickle
import spacy
import sys
import time

import numpy  as np
import pandas as pd 

from gensim.corpora           import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.matutils          import sparse2full

from sklearn.model_selection  import train_test_split

# ______________________________________________________________________________
# Process data

homework = pd.read_json('homework.json', lines=True)


downsample = False # To accelerate testing.
if downsample:
	idx = np.random.choice(np.arange(len(homework)), 1000, replace=False)
	homework = homework.iloc[idx,:]

# Clean label columns
emotion_cols = homework.columns[homework.columns.str.contains('emotion')]
for col in emotion_cols:
	homework[col] = pd.to_numeric(homework[col], errors = 'coerce')
	homework[col] = np.where(homework[col] < 0, 0, homework[col])  # Replace negative values w 0.
	homework[col].fillna(0, inplace = True)  # Replace missing values w 0.

# Clean worker_id column
homework['worker_id'] = pd.to_numeric(homework['worker_id'], errors = 'coerce')
homework['worker_id'].fillna(0, inplace = True)

# Helper saver.
def save(obj, name):
    with open('%s.pkl' % name, 'wb') as f:
        pickle.dump(obj, f)

# ______________________________________________________________________________
# TFIDF weighted word embeddings
"""
Use TFIDF to get weights for terms, and dot product word vectors for words
in text to get a 300 dimensional vector which is essentially a weighted combination
of the individual word vectors.  
"""
print('Loading `en_core_web_md`.')
nlp  = spacy.load('en_core_web_md')

# Replace '' with garbage text.
headline_for_training = homework.headline.replace('', 'MISSING HEADLINE')
summary_content_for_training = homework.summary.replace('', 'MISSING SUMMARY')
# What do we want to train on?
data = summary_content_for_training 

def keep_token(t):
    return (t.is_alpha and 
            not (t.is_space or t.is_punct or 
                 t.is_stop  or t.like_num) )

def lemmatize_doc(doc):
    return [t.lemma_ for t in doc if keep_token(t)]


s = time.time()
# Takes ~30 mins to run on full data.
print('Starting lemmatization.')
docs = [lemmatize_doc(nlp(doc)) for doc in data]

def docs_to_tfidf_embedding(docs):
	docs_dict = Dictionary(docs)
	docs_dict.filter_extremes(no_below=20, no_above=0.2)
	docs_dict.compactify()

	docs_corpus = [docs_dict.doc2bow(doc) for doc in docs]
	model_tfidf = TfidfModel(docs_corpus, id2word=docs_dict)  # Build TFIDF.
	docs_tfidf  = model_tfidf[docs_corpus]  # Apply to docs.
	docs_vecs   = np.vstack([sparse2full(c, len(docs_dict)) for c in docs_tfidf])
	try:
		tfidf_emb_vecs = np.vstack([nlp(docs_dict[i]).vector for i in range(len(docs_dict))])
	except ValueError:
		print('Maybe you need more data! Don\'t downsample.')
		sys.exit()
	docs_emb = np.dot(docs_vecs, tfidf_emb_vecs) 
	return docs_emb, docs_dict,  model_tfidf, tfidf_emb_vecs

# ______________________________________________________________________________
# Prepare for model training.

print('Convert docs to vectors.')
docs_emb, docs_dict,  model_tfidf, tfidf_emb_vecs = docs_to_tfidf_embedding(docs)
X = docs_emb
X = np.concatenate((X, np.asarray(homework.worker_id).reshape(homework.shape[0],1)), axis = 1)

print('Total data prep for {} observations took {} seconds'.
	format(homework.shape[0], time.time() - s))


# ______________________________________________________________________________
# Test train split, and file saving.

save(docs_dict, 'docs_dict')
save(model_tfidf, 'model_tfidf')
save(tfidf_emb_vecs, 'tfidf_emb_vecs')

labels = homework[emotion_cols].values

x_train, x_test, y_train, y_test = train_test_split(X, 
													labels,
													random_state = 1234)

# ______________________________________________________________________________
# Save test sets as pickled objects for model assessment post training.

save(x_train, 'x_train')
save(y_train, 'y_train')
save(x_test,  'x_test')
save(y_test,  'y_test')

