#!/usr/bin/python

import matplotlib.pyplot as plt

from sklearn import cross_validation
from sklearn import metrics
import numpy as np
import tensorflow as tf
import cv2
from matplotlib import pyplot as plt

# Load Dataset
mnist = tf.contrib.learn.datasets.load_dataset('mnist');

x_train = mnist.train.images;
y_train = mnist.train.labels;
x_test = mnist.test.images;
y_test = mnist.test.labels;

print y_train;

divided_num = 1; #5;

x_train = x_train[:int(len(x_train) / divided_num)];
y_train = y_train[:int(len(y_train) / divided_num)];
x_test = x_test[:int(len(x_test) / 1)];
y_test = y_test[:int(len(y_test) / 1)];

print "Training Size: " + str(len(x_train)); # 28 X 28 Image Set
print "Test Size: " + str(len(x_test)); # 28 X 28 Image Set

# One. File Load
def load_img(full_file_name):
    img = cv2.imread(full_file_name)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #height, width = img.shape print height, width#cv2.imwrite( file_name + "_grey.png", img )
    img = cv2.resize(img, (28, 28)) #height, width = dst.shape print height, width #plt.imshow(img) # check #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis #plt.show()
    img = (img * 1.0 - img.min()) / (img.max() - img.min())# normalize value
    #cv2.imwrite('MNIST-data/custom/save/' + file_name + "_grey.png", img )
    img = np.reshape(img, [1, 784])
    return img;
def data_sample_count(path): # for MNIST-data/English/Img/GoodImg/Bmp/
    import os
    number_count = 10;
    total_count = 0;
    for num in range(number_count):
        folder_num = num + 1
        if (num < 9) :
            folder_name = 'Sample00' + str(folder_num)
        else :
            folder_name = 'Sample0' + str(folder_num)
        
        full_folder = path + folder_name + "/";
        total_count += len(os.listdir (full_folder))
        
    return total_count;

def load_data(path): # for MNIST-data/English/Img/GoodImg/Bmp/
    import os
    import collections
    from natsort import natsorted
    
    number_count = 10;
    Dataset = collections.namedtuple('Dataset', ['data', 'target'])
    n_samples = data_sample_count(path)
    data = np.zeros((n_samples, 28 * 28), dtype=np.float)
    target = np.zeros((n_samples,), dtype=np.int)
    
    folder_names = [d for d in os.listdir (path)]; # folder name ['Sample001',...]
    folder_names = natsorted(folder_names)
    
    i = 0;
    for num in range(number_count):
        folder_num = num + 1
        if (num < 9) :
            folder_name = 'Sample00' + str(folder_num)
        else :
            folder_name = 'Sample0' + str(folder_num)
        
        full_folder = path + folder_name + "/";
        
        # number image files
        full_file_names = natsorted([full_folder + file_name for file_name in os.listdir (full_folder)]);
        for full_file_name in full_file_names:
            data[i] = np.asarray(load_img(full_file_name), dtype=np.float)
            target[i] = np.asarray(num, dtype=np.int)
            i += 1   
    return Dataset(data=data, target=target);

good_data_set = load_data("MNIST-data/English/Img/GoodImg/Bmp/"); #print len(good_data_set.data) #print len(good_data_set.target)
bad_data_set = load_data("MNIST-data/English/Img/BadImag/Bmp/");
font_data_set = load_data("MNIST-data/mac_fonts/raw_images/");

x_train = np.append(x_train, good_data_set.data, axis=0)
x_train = np.append(x_train, bad_data_set.data, axis=0)
x_train = np.append(x_train, font_data_set.data, axis=0)
y_train = np.append(y_train, good_data_set.target, axis=0)
y_train = np.append(y_train, bad_data_set.target, axis=0)
y_train = np.append(y_train, font_data_set.target, axis=0)

print "Training Size: " + str(len(x_train)); # 28 X 28 Image Set
print "Test Size: " + str(len(x_test)); # 28 X 28 Image Set

from sklearn.neighbors import KNeighborsClassifier;
from sklearn.ensemble import AdaBoostClassifier;
from sklearn.ensemble import RandomForestClassifier;
from sklearn.metrics import accuracy_score;
from time import time;

# 1. Use pure sklearn library
clfs = KNeighborsClassifier(n_neighbors=10, weights="distance"), AdaBoostClassifier(), RandomForestClassifier()

for index in range(len(clfs)):
    
    t0 = time();
    #clfs[index].fit(x_train, y_train);
    print "training time (fit):", round(time()-t0, 3), "s";
    
    t0 = time();
    #pred = clfs[index].predict(x_test);
    print "training time (predict):", round(time()-t0, 3), "s";
    
    #accuracy = accuracy_score(y_test, pred);
    
    #print accuracy;


# 2. Use tensorflow using sklearn interface

# Build 3 layer DNN with 10, 20, 10 units respectively.
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_train);
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[784, 50, 100, 500, 1000, 10, 10], n_classes=10, model_dir="save/DNNClassifier_FullDataset_Font_Mac");

# Fit and predict.
t0 = time();
#classifier.fit(x_train, y_train.astype(np.int32), batch_size=30, steps=5000); # already trained
print "training time dnn (fit):", round(time()-t0, 3), "s";

t0 = time();
predictions = list(classifier.predict(x_test, as_iterable=True));
print "training time dnn (predict):", round(time()-t0, 3), "s";

score = metrics.accuracy_score(y_test, predictions);
print('Accuracy: {0:f}'.format(score));

# 3. Test large or small image using training set (Image size normalization[scaling])
def load_imgs(dir_path, file_names):
    full_file_names = [dir_path + file_name for file_name in file_names]
    return np.array([load_img(full_file_name) for full_file_name in full_file_names]);
imgs = load_imgs('MNIST-data/custom/', ["one.jpg", "one.png", "two.jpg", "two_one.png", "three_one.jpg", "three_two.png", "four_one.jpg", "four_two.jpg", "five.jpg", "five.png", "seven_one.png", "seven_two.png"])
#print imgs[0]

preds = list(classifier.predict(imgs, as_iterable=True))
print preds
print metrics.accuracy_score([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 7, 7], preds);

