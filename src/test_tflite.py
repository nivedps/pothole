import tensorflow as tf 
import cv2
import numpy as np

TFLITE_MODEL = '../model/yolov4-416.tflite'
tflite_interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)

input_details = tflite_interpreter.get_input_details()
output_details = tflite_interpreter.get_output_details()

def get_model_details():

    print("== Input details ==")
    print("name:", input_details[0]['name'])
    print("shape:", input_details[0]['shape'])
    print("type:", input_details[0]['dtype'])

    print("\n== Output details ==")
    print("name:", output_details[0]['name'])
    print("shape:", output_details[0]['shape'])
    print("type:", output_details[0]['dtype'])


tflite_interpreter.allocate_tensors()

def testing (image_location):
    img = cv2.imread(image_location)
    new_img = cv2.resize(img, (416, 416))
    new_img = new_img.astype(np.float32)
    print(type([new_img]))

    tflite_interpreter.set_tensor(input_details[0]['index'], 
    [new_img])

    tflite_interpreter.invoke()
    rects = tflite_interpreter.get_tensor(
        output_details[0]['index'])
    print(rects)
    scores = tflite_interpreter.get_tensor(
        output_details[1]['index'])
    print(scores[0])

    height,width,c = img.shape
    print(rects.shape)
    i = 0

    for box in rects[0] :
        if np.sum(scores[0][i]) > 0.5 :
            y_min = int(max(1, (box[0] * height)))
            x_min = int(max(1, (box[1] * width)))
            y_max = int(min(height, (box[2] * height)))
            x_max = int(min(width, (box[3] * width)))

            cv2.rectangle(img, (x_min, y_min), (x_max, y_max),
            (255, 255, 255), 2)
        i+=1
    cv2.imshow('image',img)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()     


image_location = '../test/test1.jpg'
testing(image_location)