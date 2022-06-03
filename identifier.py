import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# from PIL import ImageGrab

faceID_path = 'face_ID'
images = []
idNames = []
idList = os.listdir(faceID_path)
print(idList)
for cl in idList:
    curImg = cv2.imread(f'{faceID_path}/{cl}')
    images.append(curImg)
    idNames.append(os.path.splitext(cl)[0])
print(idNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def reportNoMask(name):

    with open('report.csv', 'r+') as f:
        pictureList = []
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
            pictureList.append(entry[2])

        
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')
        f.writelines(f'{name},{dtString},{snapshot}\n')


#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr

encodeListKnown = findEncodings(images)
print('Encoding Complete')

# cap = cv2.VideoCapture(0)
snapshot_path = 'snapshot'

for snapshot in os.listdir(snapshot_path):
    imgS = cv2.imread(f'{snapshot_path}/{snapshot}')
    imgS = cv2.resize(imgS, (400, 300))
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = idNames[matchIndex].upper()
            reportNoMask(name)