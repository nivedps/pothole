''' to  move the annotation file from yolo to images folder'''

import os
import shutil

files = os.listdir("../Dataset/yolo")
print(files[0:5])
destination = "../Dataset/images"
cur_location = "../Dataset/yolo"

for f in files :
	shutil.copy(os.path.join(cur_location,f),destination)

print('copying complete !!!')	