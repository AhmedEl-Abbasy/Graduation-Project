import cv2 
import mediapipe as mp
import numpy as np
from scipy.spatial import distance

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap =cv2.VideoCapture(0)
while cap.isOpened():
    cap.set(3, 1280)
    #cap.set(4, 720)
    _, image = cap.read()
    
    results = holistic.process(image)
    face_results = results.face_landmarks
    face_landmarks_positions = []
    
    if face_results:
        for _, data_point in enumerate(face_results.landmark):
            face_landmarks_positions.append([data_point.x, data_point.y, data_point.z])
            
        face_landmarks_positions = np.array(face_landmarks_positions)
        face_landmarks_positions[:, 0] *= image.shape[1]
        face_landmarks_positions[:, 1] *= image.shape[0]
        FL=face_landmarks_positions


        A=np.array([ 
                    (int(FL[6][0]),int(FL[6][1])),
                    (int(FL[33][0]),int(FL[33][1])),            
                    (int(FL[263][0]),int(FL[263][1])),          
                    (int(FL[229][0]),int(FL[229][1])),
                    (int(FL[449][0]),int(FL[449][1])),
                    (int(FL[152][0]),int(FL[152][1]))
                     ])

        
        for p in A:
            image = cv2.circle(image, (int(p[0]),int(p[1])), 1, (255,255,255), 2)
        #cv2.line(image,(A[1][0],A[1][1]),(A[2][0],A[2][1]),(255,0,0), 2)
        #cv2.line(image,(A[3][0],A[3][1]),(A[4][0],A[4][1]),(255,0,0), 2)
        
        dist1=distance.euclidean(A[2],A[1])         
        dist2=distance.euclidean(A[3],A[4])
        dist3=distance.euclidean(A[0],A[5])
        dist=(dist1+dist2)/(2*dist3)
        
        #Acc=(dist/0.725)*100
        #cv2.putText(image,'{}'.format(int(Acc)),(30,200),16, 4,(0,128,0),5)
        if (dist<0.58) | (dist>0.9):
            cv2.putText(image,"Wroooong####",(30,400),20, 4,(0,0,255),5)
            
        
    else :
        print("no face in frame")
        cv2.putText(image,'Wrong ,No Face in Frame',(10,300),32, 4,(0,0,255),5)

    
    
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
