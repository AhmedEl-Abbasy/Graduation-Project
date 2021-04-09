import cv2
import dlib 
import numpy as np
from scipy.spatial import distance




detector =dlib.get_frontal_face_detector()
predictor =dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

thresh = 0.25
frame_check = 20



cap= cv2.VideoCapture(0)
flag=0

while True:
    _,frame =cap.read()
    gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces= detector(gray)
    for face in faces:
        #print(face)
        ##x,y=face.left(),face.top()
        ##x1,y1=face.right(),face.bottom()
        ##cv2.rectangle(frame,(x,y),(x1,y1),(0,255,0),2)
        
        landmarks=predictor(gray,face)
        ###x=landmarks.part(36).x
        ###y=landmarks.part(36).y
        #Lp=(landmarks.part(36).x,landmarks.part(36).y)
        
        
        #cv2.circle(frame,(x,y),3,(0,0,255),2)
        LeftEye=np.array([(landmarks.part(36).x,landmarks.part(36).y),
                     (landmarks.part(37).x,landmarks.part(37).y),
                     (landmarks.part(38).x,landmarks.part(38).y),
                     (landmarks.part(39).x,landmarks.part(39).y),
                     (landmarks.part(40).x,landmarks.part(40).y),
                     (landmarks.part(41).x,landmarks.part(41).y)])
    
        cv2.drawContours(frame,[LeftEye],0,(0,255,0),2)
        
        #*Right eye
        RightEye=np.array([(landmarks.part(42).x,landmarks.part(42).y),
                     (landmarks.part(43).x,landmarks.part(43).y),
                     (landmarks.part(44).x,landmarks.part(44).y),
                     (landmarks.part(45).x,landmarks.part(45).y),
                     (landmarks.part(46).x,landmarks.part(46).y),
                     (landmarks.part(47).x,landmarks.part(47).y)])
        
        cv2.drawContours(frame,[RightEye],0,(0,255,0),2)
        
        ###
        #print(distance.euclidean(Lp1,Rp1))
        leftEAR = eye_aspect_ratio(LeftEye)
        rightEAR = eye_aspect_ratio(RightEye)
        
        ear = (leftEAR + rightEAR) / 2.0
        
        if ear < thresh:
            flag += 1
            print (flag)
            if flag >= frame_check:
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10,325),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                #print ("Drowsy")
        else:
            flag = 0
    
    cv2.imshow("Frame",frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
