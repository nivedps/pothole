import os 
from zipfile import ZipFile 

#setting the directory
cur_dir = os.getcwd()
cur_dir,tail = os.path.split(cur_dir)

#zipfile path
zip_path = os.path.join(cur_dir,'archive.zip')

#creating directory
def make_dir() :
	try :
		os.mkdir(os.path.join(cur_dir,'Dataset'))
		print('Directory successfully created')
	except OSError :
		print('Directory already present')	


#extracting the zipfile
def extract_zipfile (filename,destination) :
	zip_ref = ZipFile(filename,'r')
	zip_ref.extractall(destination)

#dataset path
dataset_path = os.path.join(cur_dir,'Dataset')

make_dir()
if os.listdir(dataset_path) == 0:
	extract_zipfile(zip_path,dataset_path)	


