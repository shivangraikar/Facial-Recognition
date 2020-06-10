import cv2

face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread('abc.jpg')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
f=face.detectMultiScale(gray,scaleFactor = 1.1,minNeighbors = 7)

for x,y,w,h in f:
    img = cv2.rectangle(img,(x,y),(x+h,y+w),(0,255,0),2)
    cv2.imshow('Face recognized',img)

cv2.waitKey(0)
cv2.destroyAllWindows()
