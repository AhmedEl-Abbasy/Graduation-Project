import cv2 
import mediapipe as mp
import numpy as np
from scipy.spatial import distance

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	Ratio = (A + B) / (2.0 * C)
	return Ratio
flag=0

cap =cv2.VideoCapture(0)
while cap.isOpened():
    cap.set(3, 1280)
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
        
        ###############
        A=np.array([ 
                    (FL[6][0],FL[6][1]),
                    (FL[33][0],FL[33][1]),            
                    (FL[263][0],FL[263][1]),          
                    (FL[229][0],FL[229][1]),
                    (FL[449][0],FL[449][1]),
                    (FL[152][0],FL[152][1])
                     ])
        
        LeftEye=np.array([ 
                    (FL[33][0],FL[33][1]),
                    (FL[160][0],FL[160][1]),            
                    (FL[158][0],FL[158][1]),          
                    (FL[133][0],FL[133][1]),
                    (FL[153][0],FL[153][1]),
                    (FL[144][0],FL[144][1])
                     ])

        RightEye=np.array([ 
                    (FL[362][0],FL[362][1]),
                    (FL[385][0],FL[385][1]),            
                    (FL[387][0],FL[387][1]),          
                    (FL[263][0],FL[263][1]),
                    (FL[373][0],FL[373][1]),
                    (FL[380][0],FL[380][1])
                     ])
        
        
        for p in A:
            image = cv2.circle(image, (int(p[0]),int(p[1])), 1, (255,255,255), 2)
        for p0 in LeftEye:
            image = cv2.circle(image, (int(p0[0]),int(p0[1])), 1, (255,255,255), 2)
        for p1 in RightEye:
            image = cv2.circle(image, (int(p1[0]),int(p1[1])), 1, (255,255,255), 2)
        
        #####
        dist1=distance.euclidean(A[2],A[1])         
        dist2=distance.euclidean(A[3],A[4])
        dist3=distance.euclidean(A[0],A[5])
        dist=(dist1+dist2)/(2*dist3)
        
        ######
        LeftEyeD=eye_aspect_ratio(LeftEye)
        RightEyeD=eye_aspect_ratio(RightEye)
        
        EyeDist=(LeftEyeD+RightEyeD)/2
        
        frame_check=15       
        
        if (dist<0.58) | (dist>0.9):
            cv2.putText(image,"Wroooong####",(30,400),20, 4,(0,0,255),5)
        elif EyeDist < 0.195 :
            flag += 1
            if flag >= frame_check:
                cv2.putText(image,"####Wroooong####",(10,100),16, 4,(0,0,255),5)
                cv2.putText(image,"####Wroooong####",(10,390),16, 4,(0,0,255),5)

    else :
        cv2.putText(image,"Wrong ,No Face in Frame",(10,300),16, 4,(255,0,0),5)
    
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
