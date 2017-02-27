# https://tanmayshah2015.wordpress.com/2015/12/01/synthetic-font-dataset-generation/
from PIL import Image, ImageDraw, ImageFont
#import ttfquery.findsystem 
import string
import ntpath
import numpy as np
import os
import glob
from bokeh.core.enums import Anchor

# make font list
font_save_path = 'MNIST-data/mac_fonts/'
full_file_name = os.path.join(font_save_path, "Fonts_list.txt")
fileWrite = open(full_file_name, "w")  
files = os.listdir ('/Library/Fonts/')
for file in files:
    fileWrite.write(file + '\n');
    print file
fileWrite.close()

fontSize = 28
imgSize = (28,28)
position = (0,0)
 
#All images will be stored in 'Synthetic_dataset' directory under current directory
dataset_path = os.path.join (os.getcwd(), "MNIST-data/mac_fonts/raw_images/")
if not os.path.exists(dataset_path):
   os.makedirs(dataset_path)
 
fhandle = open(full_file_name, 'r')
lower_case_list = list(string.ascii_lowercase)
upper_case_list = list(string.ascii_uppercase)
digits = range(0,10)
 
digits_list=[]
for d in digits:
   digits_list.append(str(d))
 
all_char_list = lower_case_list + upper_case_list + digits_list
 
fonts_list = []
for line in fhandle:
   fonts_list.append(line.rstrip('\n'))
 
total_fonts = len(fonts_list)
#paths = ttfquery.findsystem.findFonts()
all_fonts = glob.glob("/Library/Fonts/*.ttf") # other extentions also need to be checked
except_fonts = ['NISC18030.ttf', 'Wingdings.ttf', 'Zapfino.ttf'];
f_flag = np.zeros(total_fonts - len(except_fonts))
 
for sys_font in all_fonts:
    
   #print "Checking "+p
   font_file = ntpath.basename(sys_font)
   font_file = font_file.rsplit('.')
   font_file = font_file[0]
   f_idx = 0
   for font in fonts_list:
      if font in except_fonts: # fonts which have some problem
          continue;
       
      f_lower = font.lower()
      s_lower = sys_font.lower()
      #Check desired font
      if f_lower in s_lower:
         path = sys_font
         print "-" + font
         font = ImageFont.truetype(path, fontSize)
         f_flag[f_idx] = 1
         for ch in all_char_list:
            image = Image.new("RGB", imgSize, (255,255,255))
            draw = ImageDraw.Draw(image)
            pos_x = 0
            pos_y = 0
            pos_idx=0
            for y in [pos_y-1, pos_y, pos_y+1]:
               for x in [pos_x-1, pos_x, pos_x+1]:
                  position = (x,y)
                  #position = (14, 14)
                  draw.text(position, ch, (0,0,0), font=font, anchor="center")
                  
                  ##without this flag, it creates 'Calibri_a.jpg' even for 'Calibri_A.jpg'
                  ##which overwrites lowercase images
                  #l_u_d_flag = "u"
                  #if ch.islower():
                  #   l_u_d_flag = "l"
                  if ch.isdigit():
                     if (int(ch) < 9) :
                        folder_name = '/Sample00' + str(int(ch) + 1)
                     else :
                        folder_name = '/Sample0' + str(int(ch) + 1)
                     full_folder_name = dataset_path + folder_name
                     if not os.path.exists(full_folder_name):
                        os.makedirs(full_folder_name)
                     
                     l_u_d_flag = "d"
                     file_name = font_file + '_' + l_u_d_flag + '_' + str(pos_idx) + '_' + ch + '.jpg'
                     file_name = os.path.join(full_folder_name,file_name)
                     image.save(file_name)
                     
                     pos_idx = pos_idx + 1
      f_idx = f_idx + 1