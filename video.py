import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np


#load the model
maskmodel=load_model("face_model.h5")
#use the camera
cap=cv2.VideoCapture(0)
#read face detector model from disk
net=cv2.dnn.readNetFromCaffe("files/deploy.prototxt","files/res10_300x300_ssd_iter_140000.caffemodel")


#index=0
while True:
    ret,frame=cap.read()
    h,w=frame.shape[:2]
    #converting the image to blob
    blob=cv2.dnn.blobFromImage(frame,1,(224,224))
    net.setInput(blob)
    output=net.forward()

    location=[]
    faces=[]

    # loop over the detections
    for i in range(0, output.shape[2]):

        # extract the confidence which will be the probability associated
        confidence = output[0, 0, i, 2]

        #chceking the threshold
        if confidence>0.45:

            #finding the coordinate of the face detected
            boxPoints = output[0, 0, i, 3:7]*np.array([w, h, w, h])
            (startX, startY, endX, endY) = boxPoints.astype("int")
            #checking if the detection is out of the frame
            (startX, startY) = (max(startX,0), max(startY,0))


            #slicing the pixels in the detected area
            faceframe=frame[startY:endY,startX:endX]
            #converting the detected area from bgr to rgb channel
            faceframe = cv2.cvtColor(faceframe, cv2.COLOR_BGR2RGB)
            #resize it to 224x224
            faceframe = cv2.resize(faceframe, (224, 224))
            faceframe=img_to_array(faceframe)
            faceframe=preprocess_input(faceframe)

            #append the values to the list
            faces.append(faceframe)
            location.append((startX, startY, endX, endY))


    if len(faces)>0:
        faces = np.array(faces, dtype="float32")
        #make prediction using the saved model
        prediction=maskmodel.predict(faces)


        for (loc,pred) in zip(location,prediction):
            (startX, startY, endX, endY) = loc
            (mask,without_mask)=pred

            #comparing the values
            if mask>without_mask:
                label="Mask"+str(round(mask*100,2))+"%"
                #displaying text
                cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_DUPLEX, 0.45, (255,255,0), 2)
                #drawing the rectangle enclosing the face on the frame
                cv2.rectangle(frame,(startX,startY),(endX,endY),(0,255,0),2)
            else:
                label="no Mask"+str(round(without_mask*100,2))+"%"
                #displaying text
                cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_DUPLEX, 0.45, (255,255,0), 2)
                #drawing the rectangle enclosing the face on the frame
                cv2.rectangle(frame,(startX,startY),(endX,endY),(0,0,255),2)

    #showing the frame
    cv2.imshow("frame",frame)
    #cv2.imwrite("videofeed"+str(index)+".png",frame)
    #index+=1

    #presss escape to break the loop
    if cv2.waitKey(1) & 0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()
