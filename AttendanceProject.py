import os
import cv2
import numpy as np
import face_recognition
from datetime import datetime

path = "ImagesAttendance"
images = []
classNames = []
mylist = os.listdir(path)


for cl in mylist:
    currImg = cv2.imread(f'{path}/{cl}')
    images.append(currImg)
    classNames.append(os.path.splitext(cl)[0])

images.pop(0)
classNames.pop(0)
def findencodings(images):
    encodelist=[]
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encoded = face_recognition.face_encodings(img)[0]
        encodelist.append(encoded)

    return encodelist

encodeListKnown = findencodings(images)

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList=f.readlines()
        nameList=[]

        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.now()
            dtString= now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')



# print(len(encodeListKnown))

print('Encoding Complete')


cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgSmall = cv2.resize(img,(0,0),None,0.25,0.25)
    imgSmall = cv2.cvtColor(imgSmall,cv2.COLOR_BGR2RGB)

    facesCurrFrame = face_recognition.face_locations(imgSmall)
    encodesCurrFrame = face_recognition.face_encodings(imgSmall,facesCurrFrame)

    for encodeFace, faceLoc in zip(encodesCurrFrame,facesCurrFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)

        # print(faceDis)
        matchIndex = np.argmin(faceDis)


        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # print(name)
            y1,x2,y2,x1=faceLoc
            y1, x2, y2, x1 =y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),2,cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            markAttendance(name)

        cv2.imshow('webcam',img)
        cv2.waitKey(1)




