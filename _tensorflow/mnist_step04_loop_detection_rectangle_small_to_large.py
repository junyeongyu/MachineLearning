from time import time
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
import tensorflow as tf
import numpy as np
import cv2
import mnist_core as core

# Use tensorflow using sklearn interface
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(np.zeros((1, 32 * 32 * 3))); #x_train
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[3072, 240, 150, 500, 1500, 8, 2], n_classes=2, model_dir="dataset/save/DNNClassifier_CIFAR100_OR_NUM");

# 1. load original image.
full_file_name = "dataset/test-images/love-number-640x1024.jpg"
img = cv2.imread(full_file_name)
img_height = img.shape[0];
img_width= img.shape[1];
img_min_height_width = min(img_height, img_width);

# 2. decide rectangle size and step
# start rectangle size 4 * 4 (2 step size--> half of height of width) (every loop -> size 4 increase), end rectangle size 640 * 640
#  loop: until rect size < min(width of original image, height of original image)
#      loop: up to down step count = (height of original image - (hegiht of rect image - step size)) / step size
#          loop: left to right step count = (width of original image - (width of rect image - step size)) / step size
#  total step count = rect size change count * left to right step count * up to down step count
t0 = time();
total_step_count = 0; # it is necessary to define for making np result array
rect_size = 40.0
while rect_size < img_min_height_width: # later will find formula instead of using loop
    rect_size = int(rect_size * 1.5);
    step_size = int(rect_size / 4)
    step_count_up_to_down = int((img_height - (rect_size - step_size)) / step_size)
    step_count_left_to_right = int((img_width - (rect_size - step_size)) / step_size)
    total_step_count += step_count_left_to_right * step_count_up_to_down

rect_size = 40.0 # intialize again
img_rects = np.zeros((total_step_count, 32 * 32 * 3)) # because result of normalize is 32 * 32 * 3.
img_rect_metas = []
current_step_count = 0
while rect_size < img_min_height_width:
    #rect_size += 4; // performance issue
    rect_size = int(rect_size * 1.5); # *= 1.5 => 2 mins for 128759 (total_step_count)
    step_size = int(rect_size / 4)
    step_count_up_to_down = int((img_height - (rect_size - step_size)) / step_size)
    step_count_left_to_right = int((img_width - (rect_size - step_size)) / step_size)
    #print 'step_count_left_to_right: ', step_count_left_to_right
    
    for i in range(step_count_up_to_down):
        for j in range(step_count_left_to_right):
            # 3. normalize original image of only rect range 
            x = j * step_size
            y = i * step_size
            img_rect = img[y:y+rect_size, x:x+rect_size] # raw rect img (not normalized)
            img_rect = core.normalize(img_rect) # normalize rect img
            img_rects[current_step_count] = img_rect
            img_rect_metas.append({'x': x, 'y': y, 'size': rect_size});
            #cv2.rectangle(img, (x,y),(x+int(rect_size),y+int(rect_size)),(0,255,0),1) # for test
            current_step_count += 1;
            #break # for test
        #break # for test
    
    #break # for test
print "rect images loop time dnn (", current_step_count,") :", round(time()-t0, 3), "s";

#cv2.imwrite('dataset/test-images/love-number-640x1024-number-detection.jpg', img) # for test


# 4. predict all rect images based on ML Algorithm (1 -> number, 0 -> not number)
t0 = time();
predictions = list(classifier.predict(img_rects));
print "predict time dnn (", current_step_count,") :", round(time()-t0, 3), "s";
#print predictions
print np.sum(predictions);

# 5. make target image & draw rectangle for only number part --> for human checking
for i in range(total_step_count): # Need to find better way to improve performance (using vector calculation)
    #img_rect = img_rects[i]
    pred = predictions[i]
    if (pred == 1): # for only number
        img_rect_mata = img_rect_metas[i]
        x = img_rect_mata['x']
        y = img_rect_mata['y']
        size = img_rect_mata['size']
        #print x, ",", y, ",", size
        cv2.rectangle(img, (x,y),(x+size,y+size),(0,(i * 1.0 / total_step_count) * 255,(1 - i * 1.0 / total_step_count) * 255),1) #3
cv2.imwrite('dataset/test-images/love-number-640x1024-number-detection-small-to-large.jpg', img)
#cv2.imshow('Number Detection', img) # not supported


