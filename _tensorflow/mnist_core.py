import numpy as np
import tensorflow as tf
import cv2
import os

def normalize(img):
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #height, width = img.shape print height, width#cv2.imwrite( file_name + "_grey.png", img )
    img = cv2.resize(img, (32, 32)) #height, width = dst.shape print height, width #plt.imshow(img) # check #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis #plt.show()
    img = (img * 1.0 - img.min()) / (img.max() - img.min())# normalize value
    #cv2.imwrite('MNIST-data/custom/save/' + file_name + "_grey.png", img )
    img = np.reshape(img, [1, 32 * 32 * 3])
    return img

def load_partial_imgs(dir_name):
    #print os.listdir(dir_name)
    total_step_count = 0
    divided_count = 16
    while divided_count >= 1:
        total_step_count += (divided_count * divided_count)
        divided_count /= 2
    
    total_step_count *= len(os.listdir(dir_name))
    
    print total_step_count
    
    img_rects = np.zeros((total_step_count, 32 * 32 * 3))
    
    current_step_count = 0
    for file_name in os.listdir(dir_name):
        full_file_name = dir_name + file_name
        img = cv2.imread(full_file_name)
        img_height = img.shape[0]
        img_width= img.shape[1]
        
        divided_count = 16;
        
        while divided_count >= 1:
            rect_height = img_height / divided_count
            rect_width = img_width / divided_count
            for i in range(divided_count):
                for j in range(divided_count):
                    x = j * rect_width
                    y = i * rect_height
                    
                    img_rect = img[y:y+rect_height, x:x+rect_width] # raw rect img (not normalized)
                    img_rect = normalize(img_rect) # normalize rect img
                    img_rects[current_step_count] = img_rect
                    
                    #if i == divided_count - 1 and j == divided_count - 1:
                        
                    
                    current_step_count += 1;
            divided_count /= 2;

        print file_name

## Test
#load_partial_imgs('dataset/scene-images/')


def load_img(full_file_name):
    img = cv2.imread(full_file_name)
    img = normalize(img)
    return img;
def data_sample_count(path): # for MNIST-data/English/Img/GoodImg/Bmp/
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
    data = np.zeros((n_samples, 32 * 32 * 3), dtype=np.float)
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