# machine-learning-notebook

![#c5f015](https://placehold.it/15/c5f015/000000?text=+) `http://scikit-learn.org/` 

```
Scikit makes Machine Learning Easy using Python
```
![#c5f015](https://placehold.it/15/c5f015/000000?text=+) `Naive Bayes` 

```
A statistical approach to predict the nature of data you have
```

![#c5f015](https://placehold.it/15/c5f015/000000?text=+) `My way of defining FEATURES` 
```
 In a GRAPH , the co-ordinates (1,2) (-9,7) (-2,-7) (5,4) are FEATURES
```
![#c5f015](https://placehold.it/15/c5f015/000000?text=+) `My way of defining LABELS` 
```
 In a GRAPH , the co-ordinates (1,2) (-9,7) (-2,-7) (5,4) are FEATURES and if I group 
 (-9,7) (-2,-7) as `NEGATIVES` or `1` and (1,2) (5,4) as `POSTIVES` or `2` it means that 
 I just LABELLED my FEATURES
```

## GaussianNB

  A classfier of type `GaussianNB` is used to `fit` or `train`. Its given a sample of FEATURES and their corresponding LABELS.
  Once this is done the classifier is called `Trained Classifier`
  
  ```python
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

  ```
