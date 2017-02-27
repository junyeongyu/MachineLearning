from time import time
from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np
import cv2

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

data_train_cifar = unpickle("dataset/cifar-100/train");
data_test_cifar = unpickle("dataset/cifar-100/test");

# regulize
def regulize(x):
    x_max = np.amax(x, axis=1)
    x_min = np.amin(x, axis=1)
    x = np.divide(x.transpose() * 1.0 - x_min, (x_max - x_min)).transpose()
    return x

# 1. load raw data
x_train = regulize(data_train_cifar['data']);
y_train = np.array(data_train_cifar['coarse_labels']); #fine_labels
x_test = regulize(data_test_cifar['data']);
y_test = np.array(data_test_cifar['coarse_labels']); #fine_labels

# 2. Use tensorflow using sklearn interface

# Build 3 layer DNN with 10, 20, 10 units respectively.
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_train);
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[1024, 80, 150, 500, 1500, 20, 20], n_classes=20, model_dir="dataset/save/DNNClassifier_CIFAR100");

# Fit and predict.
t0 = time();
classifier.fit(x_train, y_train.astype(np.int32), batch_size=30000, steps=50); # already trained
print "training time dnn (fit):", round(time()-t0, 3), "s";

t0 = time();
predictions = list(classifier.predict(x_test, as_iterable=True));
print "training time dnn (predict):", round(time()-t0, 3), "s";

score = accuracy_score(y_test, predictions);
print('Accuracy: {0:f}'.format(score));
