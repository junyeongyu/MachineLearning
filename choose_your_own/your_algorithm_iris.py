#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

from sklearn import cross_validation
import numpy as np
import tensorflow as tf

# Load Dataset
#features_train, labels_train, features_test, labels_test = makeTerrainData()

iris = tf.contrib.learn.datasets.load_dataset('iris')
x_train, x_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

features_train = x_train;
labels_train = y_train;
features_test = x_test;
labels_test = y_test;

### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
#grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
#bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
#grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
#bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
#plt.xlim(0.0, 1.0)
#plt.ylim(0.0, 1.0)
#plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
#plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
#plt.legend()
#plt.xlabel("bumpiness")
#plt.ylabel("grade")
#plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.ensemble import AdaBoostClassifier;
from sklearn.ensemble import RandomForestClassifier;
from sklearn.metrics import accuracy_score;
from time import time;

# 1. Use pure sklearn library
clfs = KNeighborsClassifier(n_neighbors=10, weights="distance"), AdaBoostClassifier(), RandomForestClassifier()

for index in range(len(clfs)):
    
    t0 = time();
    clfs[index].fit(features_train, labels_train);
    print "training time (fit):", round(time()-t0, 3), "s";
    
    t0 = time();
    pred = clfs[index].predict(features_test);
    print "training time (predict):", round(time()-t0, 3), "s";
    
    accuracy = accuracy_score(labels_test, pred);
    
    print accuracy;

    #try:
    #    prettyPicture(clfs[index], features_test, labels_test)
    #except NameError:
    #    pass

# 2. Use tensorflow using sklearn interface
#from sklearn import cross_validation
from sklearn import metrics
#import numpy as np
#import tensorflow as tf

# Load dataset.
#iris = tf.contrib.learn.datasets.load_dataset('iris')
#x_train, x_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

#x_train = np.array(features_train);
#y_train = np.array(labels_train, dtype=np.int);
#x_test = np.array(features_test);
#y_test = np.array(labels_test, dtype=np.int);

# Build 3 layer DNN with 10, 20, 10 units respectively.
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_train);
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=3);

# Fit and predict.
t0 = time();
classifier.fit(x_train, y_train, steps=200);
print "training time dnn (fit):", round(time()-t0, 3), "s";

t0 = time();
predictions = list(classifier.predict(x_test, as_iterable=True));
print "training time dnn (predict):", round(time()-t0, 3), "s";

score = metrics.accuracy_score(y_test, predictions);
print('Accuracy: {0:f}'.format(score));