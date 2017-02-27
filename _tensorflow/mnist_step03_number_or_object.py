from time import time
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
import tensorflow as tf
import numpy as np
import cv2
import mnist_core as core
#sys.path.append("../tools/")


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

# 1. load cifar & number data
#mnist = tf.contrib.learn.datasets.load_dataset('mnist');
good_data_set = core.load_data("MNIST-data/English/Img/GoodImg/Bmp/"); #print len(good_data_set.data) #print len(good_data_set.target)
bad_data_set = core.load_data("MNIST-data/English/Img/BadImag/Bmp/");
font_data_set = core.load_data("MNIST-data/mac_fonts/raw_images/");

# 2. Make test sets using font_data_set
x_train_font, x_test_font, y_train_font, y_test_font = cross_validation.train_test_split(font_data_set.data, font_data_set.target, test_size=0.2, random_state=42)

#x_train = mnist.train.images
x_train = good_data_set.data
x_train = np.append(x_train, bad_data_set.data, axis=0)
x_train = np.append(x_train, x_train_font, axis=0)
x_train = np.append(x_train, regulize(data_train_cifar['data']), axis=0);

#y_train = mnist.train.labels
y_train = good_data_set.target
y_train = np.append(y_train, bad_data_set.target, axis=0)
y_train = np.append(y_train, y_train_font, axis=0)
y_train = np.full((len(y_train), 1), 1, dtype=np.int) # make number as 1
y_train = np.append(y_train, np.full((len(np.array(data_train_cifar['coarse_labels'])), 1), 0, dtype=np.int), axis=0) # make object as 0

x_test = regulize(data_test_cifar['data']);
x_test = np.append(x_test, x_test_font, axis=0)
y_test = np.full((len(np.array(data_test_cifar['coarse_labels'])), 1), 0, dtype=np.int); #fine_labels # make object as 0
y_test = np.append(y_test, np.full((len(y_test_font), 1), 1, dtype=np.int), axis=0) # make number as 1

print len(x_train)
print len(x_test)

# 3. Use tensorflow using sklearn interface
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_train);
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[3072, 240, 150, 500, 1500, 8, 2], n_classes=2, model_dir="dataset/save/DNNClassifier_CIFAR100_OR_NUM_XXX");

# Fit and predict.
t0 = time();
classifier.fit(x_train, y_train.astype(np.int32), batch_size=30000, steps=200); # already trained
print "training time dnn (fit):", round(time()-t0, 3), "s";

t0 = time();
predictions = list(classifier.predict(x_test, as_iterable=True));
print "training time dnn (predict):", round(time()-t0, 3), "s";

score = accuracy_score(y_test, predictions);
print('Accuracy: {0:f}'.format(score));
