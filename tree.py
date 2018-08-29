#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" tree.py

	A module to train a multi-label [DecisionTreeClassifier](http://scikit-learn.org/stable/modules/multiclass.html).

"""

# ______________________________________________________________________________
# Imports

import pandas 
import pickle
import time

from datetime     import datetime
from sklearn.tree import DecisionTreeClassifier

# ______________________________________________________________________________
# Load data.

try:
	x_train = pickle.load(open("x_train.pkl", "rb"))
	y_train = pickle.load(open("y_train.pkl", "rb"))
	x_test  = pickle.load(open("x_test.pkl", "rb"))
	y_test  = pickle.load(open("y_test.pkl", "rb"))
except FileNotFoundError:
	from data_prep import x_train, x_test, y_train, y_test

# ______________________________________________________________________________
# Time it.

s = time.time()
print('Starting fit.')

# ______________________________________________________________________________
# Training.

def make_model(x_train, y_train):
	clf = DecisionTreeClassifier()
	clf.fit(X = x_train, y = y_train)
	return clf
clf = make_model(x_train, y_train)
print('Training took {} seconds'.format(time.time() - s))

# ______________________________________________________________________________
# Save model as pkl file.

date_str = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
filename = 'tree_model-{}.pkl'.format(date_str)

with open('{}'.format(filename), 'wb') as f:
	pickle.dump(clf, f)