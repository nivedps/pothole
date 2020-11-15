import cv2
import numpy as np
import argparse

#defining the argument parser
parser = argparse.ArgumentParser(description="Pothole Detection")
parser.add_argument("--image",type=str,required=False,help="image location")
args = vars(parser.parse_args())

#loading the yolo weights
config_location = "../model/yolov4-tiny-custom.cfg"
weights_location = "../model/yolov4-tiny-custom_2000.weights"
net = cv2.dnn.readNet(weights_location,config_location)

#setting the colour
color = (230,230,230)

#setting the classes
classes = ["Pothole"]

#getting the output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#reading the image
img = cv2.imread(args['image'])
org_img = img
org_h,org_w,org_c = org_img.shape
# img = cv2.resize(img,None,fx=0.4,fy=0.4)

#reading and resizeing image
def image_detection(img):
    blob = cv2.dnn.blobFromImage(img, 0.008, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    output = net.forward(output_layers)
    return output

#marking the potholes
def label_image(detection,confidence,index,img):
    height,width,c = img.shape
    center_x = int(detection[0] * width)#*org_w
    center_y = int(detection[1] * height)#*org_h
    w = int(detection[2] * width)#*org_w
    h = int(detection[3] * height)#*org_h
    x = int(center_x - w / 2)
    y = int(center_y - h / 2)
    label = str(classes[index])
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, label, (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)


#finding all potholes in an image
def potholes(output,img):
    for i in output:
        for detection in i:
            score = detection[5:]
            index = np.argmax(score)
            confidence = score[index]
            if confidence > 0.3 :
                label_image(detection,confidence,index,img)


output = image_detection(img)
potholes(output,img)
cv2.imshow('Image',img)
key = cv2.waitKey(0)
cv2.destroyAllWindows()







