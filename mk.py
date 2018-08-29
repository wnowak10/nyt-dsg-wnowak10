#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" mk.py
    Usage:

    1. Fully re process data, train model (TIME INTENSIVE):
        $ python mk.py --preprocess_data True --do_train True

    2. (Recommended) Use pre-saved processed data, model:
        $ python mk.py

    Both options will:

    1. Print to stderr or a file, an  appropriate quantitative estimate 
    of the model's ability to correctly predict emotional reactions.
"""

# ______________________________________________________________________________
# Imports

from __future__ import print_function

import argparse
import sys
import pickle

FLAGS = None

def get_data():
  if FLAGS.preprocess_data:
    print('To do: preprocess data.')
    from data_prep import x_train, x_test, y_train, y_test, docs_dict, model_tfidf, tfidf_emb_vecs
    print('Data preprocessed and imported.')
    return x_train, x_test, y_train, y_test, docs_dict, model_tfidf, tfidf_emb_vecs
  else:
    print('To do: simply load pickled data.')
    try:
      x_train = pickle.load(open("x_train.pkl", "rb"))
      y_train = pickle.load(open("y_train.pkl", "rb"))
      x_test  = pickle.load(open("x_test.pkl", "rb"))
      y_test  = pickle.load(open("y_test.pkl", "rb"))

      docs_dict      = pickle.load(open("docs_dict.pkl", "rb"))
      model_tfidf    = pickle.load(open("model_tfidf.pkl", "rb"))
      tfidf_emb_vecs = pickle.load(open("tfidf_emb_vecs.pkl", "rb"))

      print('Data loaded.')
    except FileNotFoundError:
      print('You don\'t seem to have access to all the training and test data. Please recall with `--preprocess_data`.')
      sys.exit()
    return x_train, x_test, y_train, y_test, docs_dict, model_tfidf, tfidf_emb_vecs


def train_or_load_model():
    """
    Options:

    1. If --do_train true, we train and use that model.
    2. If --do_train option not provided, or set to false, we use default saved model:    
        `default_model.pkl`
    3. Unless --load_model_from path is provided, in which case we load from there.
    """
    if FLAGS.do_train:
      print('OK, we\'ll retrain the model. Here we go...')
      import tree
      model = tree.make_model(x_train, y_train)
      print('Training done.')
    else:
      print('Loading pickled model.')
      model_path = FLAGS.load_model_from  # Load default model if no model path provided.
      with open(model_path, 'rb') as f:
        model = pickle.load(f)
      print('Loaded pickled model.')
    return model

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def print_hamming_loss(model, x_test):
    """
    Hamming loss as metric for multi-label clasification:

    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.hamming_loss.html

    Simple, but must be considered against a naive baseline.
    """
    y_pred = model.predict(x_test)
    from sklearn.metrics import hamming_loss
    if FLAGS.verbose:
        print('\n --> Hamming loss for a randomly held out set is {}.\n'.format(hamming_loss(y_test, y_pred)))
    return hamming_loss(y_test, y_pred)

# ______________________________________________________________________________
# Imports

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
      '--load_model_from',
      type=str,
      default='default_model.pkl',
      help="""\
      Path to pickled prediction model.
      """)

    parser.add_argument(
      '--do_train',
      default=False,
      action ="store_true",
      help="""\
      Boolean - would you like to train a model? Warning -- this increases run time 
      significantly.
      """)

    parser.add_argument(
      '--preprocess_data',
      default=False,
      action ="store_true",
      help="""\
      Boolean - would you like to re preprocess all input and output data?
      Warning -- this increases run time significantly.
      """)

    parser.add_argument(
      '--verbose',
      default=True,
      action ="store_true",
      help="""\
      Control verbosity of execution.
      """)

    FLAGS, unparsed = parser.parse_known_args()
    x_train, x_test, y_train, y_test, docs_dict, model_tfidf, tfidf_emb_vecs = get_data()
    model = train_or_load_model() 
    eprint(print_hamming_loss(model, x_test))  # Requires x_test, y_test data to be loaded from pickle or generated from script.
    import serve
    serve.main(model, docs_dict, model_tfidf, tfidf_emb_vecs)  