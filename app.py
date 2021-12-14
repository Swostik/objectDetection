import streamlit as st
import cv2

import numpy as np
import time
from PIL import Image


st.title("Object Detection")
st.sidebar.markdown("# Model")
confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)


net = cv2.dnn.readNetFromDarknet('yolov4-tiny-custom.cfg', 'yolov4-tiny-custom_last.weights')
classes = ['headphone', 'wallet']
layer_names = net.getLayerNames()
print(layer_names)
print(net.getUnconnectedOutLayers())

outputlayers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0,255,size=(len(classes),3))
run=st.checkbox('Open/Close your Webcam')
video = cv2.VideoCapture(0)

 
starting_time = time.time()
frame_id = 0
font = cv2.FONT_HERSHEY_PLAIN


frame_window = st.image([])
frame_text = st.markdown("object is missing")
frame_text2 = st.markdown("")
while run:
    _,frame=video.read()
    #a = frame_window.image(frame)
    #st.write(a)

    height, width, channels = frame.shape
    # detecting objects
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0,0,0), swapRB =True, crop = False)
    net.setInput(blob)
    outs = net.forward(outputlayers)
    boxes = []
    confidences = []
    class_ids = []
    for output in outs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > .6:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold,.4)
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x,y), (x+w, y+h), color,2)
            cv2.putText(frame, label + ' ' + str(round(confidence,2)), (x,y+30), font, 1, (255,255,255),2)
            frame_text.markdown(label+" is found")
        else:
            frame_text.markdown(str(classes[class_ids[i]])+" is not found")
    elapsed_time = time.time() - starting_time
    fps=frame_id/elapsed_time
    #code to change the blue intensive video to normal
    img_np = np.array(frame)
    frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    #adding the frame to above created empty frame_window
    frame1 = cv2.putText(frame, 'FPS:'+str(round(fps,2)), (10,50), font, 2, (0,0,0),1)
    frame_window.image(frame1)
    #run=st.checkbox("Close", value = False)
    
    #class_name = ([classes[x] for x in class_ids])
    #i=0
    #while i<2:
        #if classes[i] in class_name:
            #frame_text.markdown(classes[i]+" is found")
        #else:
            #frame_text.markdown(classes[i]+" is not found")
        #i +=1

    

