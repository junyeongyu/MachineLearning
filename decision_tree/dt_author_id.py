#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score;

#features_train = features_train[:len(features_train)/100];
#labels_train = labels_train[:len(labels_train)/100];

clf = DecisionTreeClassifier(min_samples_split=40);

print len(features_train[0]);

t0 = time();
clf.fit(features_train, labels_train);
print "training time (fit):", round(time()-t0, 3), "s";

t0 = time();
pred = clf.predict(features_test);
print "training time (predict):", round(time()-t0, 3), "s";

accuracy = accuracy_score(labels_test, pred);

print accuracy;



#########################################################


