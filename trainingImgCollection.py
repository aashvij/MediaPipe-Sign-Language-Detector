import cv2
import os
import time
import uuid

IMAGES_PATH = "aashvijain/projects/typescriptFirst/trainingImages/"

labels = ['A', 'B', 'C']
numberOfImgs = 15

for label in labels:
    #make folder for each label
    os.path.join(IMAGES_PATH, label)
    
    #start video capture using opencv
    cap = cv2.VideoCapture(1)
    print('Collecting images for ')

    #5 second delay
    time.sleep(5)

    #loop through to take all images
    for imgNum in range(numberOfImgs):
        ret, frame = cap.read()
        #name the image with the label name and time stamp
        imageName = os.path.join(IMAGES_PATH, label, label+"."+'{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(imageName, frame)
        cv2.imshow('frame', frame)
        time.sleep(2)

        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
    cap.release()