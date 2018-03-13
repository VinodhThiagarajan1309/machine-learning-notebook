#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
# Defining an array of FEATURES ( here in this case they are graph co-ordinates )
features = np.array([
    [-1, -1],
    [-2, -1],
    [-3, -2],
    [1, 1],
    [2, 1],
    [3, 2],
    ])
# Matching labels to each of the previously defined graph co-ordinates. Both array
# size must matter
labels = np.array([
    1,
    1,
    1,
    2,
    2,
    2,
    ])

# Import Classifier - here we are using GaussianNB
from sklearn.naive_bayes import GaussianNB
# Create the classifier clf
clf = GaussianNB()
# Provide this mapping to the classifier to train itself
clf.fit(features, labels)
# Now call the predict method by passing in sample to the classifiers
# Here we are passing 6 samples
pred =  clf.predict([[1, 1],[-1, -1],[-1, -1],[-1, -1],[-1, -1],[-1, -1]])

# Let us see how accurate is our model
from sklearn.metrics import accuracy_score

# We are calling the accuracy score method and providing our prediction results
# and actual results as inputs to get a score. Below is a 100% accuracy
# to fail it change the 2 or 1 to something else
print(accuracy_score(pred,(2,1,1,1,1,1)))
