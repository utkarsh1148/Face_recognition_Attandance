import numpy as np
import cv2
import csv
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create();
rec.read("trainingdata.yml")
id=0
person=[]


def firstDigit(n):
    while n>=10:
        n=n/10
    return int(n)

def mode(List): 
    counter = 0
    num = List[0] 
      
    for i in List: 
        curr_frequency = List.count(i) 
        if(curr_frequency> counter): 
            counter = curr_frequency 
            num = i 
  
    return num


font=cv2.FONT_HERSHEY_SIMPLEX




while (1):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        print(conf)
        id=firstDigit(id)
        person.append(id)
        if (int(id)==2):
            id = 'Utkarsh'
        else:
            if (int(id)==5):
                id = 'Ramit'
            else:
                if (int(id)==1):
                    id="Prabhav"
        
            
        
        cv2.putText(img,str(id),(x,y+h),font,2.0,(255, 255, 00))
        cv2.imshow('img',img)
    
    if cv2.waitKey(1) == ord('q'):
        break


mark=mode(person)
print(mark)



cap.release()

cv2.destroyAllWindows()

print("hi")


