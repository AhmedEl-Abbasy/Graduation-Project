#pip install imutils
#!pip install mediapipe opencv2-python

import mediapipe as mp
import cv2 
from scipy.spatial import distance
import dlib
import numpy as np


detector =dlib.get_frontal_face_detector()
predictor =dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
	
thresh = 0.25
frame_check = 15    	# Time needed to show an alert

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)
flag=0

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        #frame = imutils.resize(frame, width=450)
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Make Detections
        results = holistic.process(image)
        # print(results.face_landmarks)
        
        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
        
        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        faces= detector(image)
        subjects = detector(image, 0)
        # 1. Draw face landmarks
        """mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
        """
        for face in subjects:
            landmarks=predictor(image,face)
            #shape = face_utils.shape_to_np(shape)
            
            LeftEye=np.array([(landmarks.part(36).x,landmarks.part(36).y),
                             (landmarks.part(37).x,landmarks.part(37).y),
                             (landmarks.part(38).x,landmarks.part(38).y),
                             (landmarks.part(39).x,landmarks.part(39).y),
                             (landmarks.part(40).x,landmarks.part(40).y),
                             (landmarks.part(41).x,landmarks.part(41).y)])
    
            cv2.drawContours(frame,[LeftEye],0,(0,255,0),2)
    
    
            RightEye=np.array([(landmarks.part(42).x,landmarks.part(42).y),
                     (landmarks.part(43).x,landmarks.part(43).y),
                     (landmarks.part(44).x,landmarks.part(44).y),
                     (landmarks.part(45).x,landmarks.part(45).y),
                     (landmarks.part(46).x,landmarks.part(46).y),
                     (landmarks.part(47).x,landmarks.part(47).y)])
        
            cv2.drawContours(frame,[RightEye],0,(0,255,0),2)
        
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
            
        # 2. Right hand
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )

        # 3. Left Hand
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
                        
        cv2.imshow('Raw Webcam Feed', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()