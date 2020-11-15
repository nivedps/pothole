import os
import xml.etree.ElementTree as et 

annotation_dir = '../Dataset/annotations'
yolo_dir = '../Dataset/yolo'

def make_dir():
	try :
		os.mkdir(yolo_dir)
		print('Directory created')
	except OSError :
		print('Directory exists')


make_dir()

for fp in os.listdir(annotation_dir) :

	root = et.parse(os.path.join(annotation_dir,fp)).getroot()
	xmin,ymin,xmax,ymax = 0,0,0,0
	size = root.find('size')
	width = float(size[0].text)
	height = float(size[1].text)
	filename = root.find('filename').text

	for child in root.findall('object'):
		sub = child.find('bndbox')
		label = child.find('name').text
		xmin = float(sub[0].text)
		ymin = float(sub[1].text)
		xmax = float(sub[2].text)
		ymax = float(sub[3].text)

		x_center = (xmin + xmax)/(2*width)
		y_center = (ymin + ymax)/(2*height)
		w = (xmax - xmin)/width
		h = (ymax - ymin)/height

		with open(os.path.join(yolo_dir,fp.split('.')[0] + '.txt'),'a+') as f :
			f.write(' '.join([str(label),str(x_center),str(y_center),str(w),str(h) + '\n']))


print('created yolo file')			





