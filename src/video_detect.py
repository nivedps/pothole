import cv2
import numpy as np
import argparse
import requests

#defining the argument parser
parser = argparse.ArgumentParser(description="pothole detector")
parser.add_argument("--video",type=str,required=False,help="video location")
args = vars(parser.parse_args())

#loading the yolo weights and configuartion file
config_location = "../model/yolov4-tiny-custom.cfg"
weights_location = "../model/yolov4-tiny-custom_2000.weights"
net = cv2.dnn.readNet(weights_location,config_location)

#setting the colour
color = (0,0,255)

#setting the classes
classes = ["Pothole"]

#getting the output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


#reading and resizeing image
def image_detection(img):
    blob = cv2.dnn.blobFromImage(img, 0.005, (416, 416), (0, 0, 0), True, crop=False)
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

#getting all the frames using webcam
def webcam():
    display = True
    while display :
        ret,frame = cap.read()
        output = image_detection(frame)
        potholes(output,frame)
        cv2.imshow("Frame",frame)
        if cv2.waitKey(1) & 0xFF == ord('q') :
            display = False

    cap.release()
    cv2.destroyAllWindows()   

#getting all frames using mobile cam
def mobile_cam():

    display = True
    while display :
        img_res = requests.get("http://192.168.43.1:8080/shot.jpg?rnd=158166")
        img_arr = np.array(bytearray(img_res.content), dtype = np.uint8)
        img = cv2.imdecode(img_arr,-1)
        output = image_detection(img)
        potholes(output,img)
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            display = False

    cv2.destroyAllWindows()        


if __name__ == "__main__" :

    if args['video'] :
        cap = cv2.VideoCapture(args['video'])
    else :
        cap = cv2.VideoCapture(0)
    webcam()
    # mobile_cam()
