import cv2
from time import sleep
import datetime

cap = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier('cascade.xml')

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(gray,
                                     scaleFactor=1.1,
                                     minNeighbors=10,
                                     minSize=(50,50))

    if len(faces) != 0:
        now=datetime.datetime.now()
        nowf=now.strftime('%Y-%m-%d_%H-%M-%S')
        file_name= nowf + '.png'
        cv2.imwrite(file_name, frame)
        sleep(5)

cap.release()
    


