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
full_file_name = "dataset/test-images/sign-number-001.jpg"
img = cv2.imread(full_file_name)
img_height = img.shape[0];
img_width= img.shape[1];
img_min_height_width = min(img_height, img_width)

detected_area = np.zeros((img_height, img_width))

# 2. decide rectangle size and step
# start rectangle size 4 * 4 (2 step size--> half of height of width) (every loop -> size 4 increase), end rectangle size 640 * 640
#  loop: until rect size < min(width of original image, height of original image)
#      loop: up to down step count = (height of original image - (hegiht of rect image - step size)) / step size
#          loop: left to right step count = (width of original image - (width of rect image - step size)) / step size
#  total step count = rect size change count * left to right step count * up to down step count
t0 = time();
total_step_count = 0; # it is necessary to define for making np result array
min_rect_size =  40.0;
rect_size = img_min_height_width
while rect_size > min_rect_size: # later will find formula instead of using loop
    step_size = int(rect_size / 6)
    step_count_up_to_down = int((img_height - (rect_size - step_size)) / step_size)
    step_count_left_to_right = int((img_width - (rect_size - step_size)) / step_size)
    total_step_count += step_count_left_to_right * step_count_up_to_down
    rect_size = int(rect_size / 1.5)

rect_size = img_min_height_width # intialize again
img_rects = np.zeros((total_step_count, 32 * 32 * 3)) # because result of normalize is 32 * 32 * 3.
img_rect_metas = []
current_step_count = 0
while rect_size > min_rect_size:
    step_size = int(rect_size / 6)
    step_count_up_to_down = int((img_height - (rect_size - step_size)) / step_size)
    step_count_left_to_right = int((img_width - (rect_size - step_size)) / step_size)
    
    for i in range(step_count_up_to_down):
        for j in range(step_count_left_to_right):
            # 3. normalize original image of only rect range 
            x = j * step_size
            y = i * step_size
            img_rect = img[y:y+rect_size, x:x+rect_size] # raw rect img (not normalized)
            img_rect = core.normalize(img_rect) # normalize rect img
            img_rects[current_step_count] = img_rect
            img_rect_metas.append({'x': x, 'y': y, 'size': rect_size});
            current_step_count += 1;
    rect_size = int(rect_size / 1.5)

print "rect images loop time dnn (", current_step_count,") :", round(time()-t0, 3), "s";

# 4. predict all rect images based on ML Algorithm (1 -> number, 0 -> not number)
t0 = time();
predictions = list(classifier.predict(img_rects));
print "predict time dnn (", current_step_count,") :", round(time()-t0, 3), "s";
print np.sum(predictions);

for i in range(total_step_count): # Need to find better way to improve performance (using vector calculation)
    pred = predictions[i]
    if (pred == 1): # for only number
        img_rect_mata = img_rect_metas[i]
        x = img_rect_mata['x']
        y = img_rect_mata['y']
        size = img_rect_mata['size']
        
        # 5. When number is detected, do not check duplicate number again (when duplicated area rate is more than 30% should not check again.)
        rect_in_detected_area = detected_area[y:y+size,x:x+size]
        collision_ratio = np.sum(rect_in_detected_area) * 1.0 / (size * size)
        detected_area[y:y+size,x:x+size] = 1
        
        # 6. make target image & draw rectangle for only number part --> for human checking
        if collision_ratio < 1.1:
            cv2.rectangle(img, (x,y),(x+size,y+size),(0,(i * 1.0 / total_step_count) * 255,0),1) #3
cv2.imwrite('dataset/test-images/sign-number-001-number-detection.jpg', img)