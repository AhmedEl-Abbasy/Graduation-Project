import mediapipe as mp
import cv2 as cv 
from scipy.spatial import distance
import imutils
from imutils import face_utils
import dlib


def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
	
thresh = 0.25
frame_check = 15    	# Time needed to show an alert

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# Get the left and Right Eyes indices in order to pass calculations
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]



mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


cap = cv.VideoCapture(0)
flag=0

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=450)
        
        # Recolor Feed
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # Make Detections
        results = holistic.process(image)
        # print(results.face_landmarks)
        
        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
        
        # Recolor image back to BGR for rendering
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        subjects = detect(image, 0)
        # 1. Draw face landmarks
        """mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
        """
        for subject in subjects:
            shape = predict(image, subject)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            
            ear = (leftEAR + rightEAR) / 2.0
            
            leftEyeHull = cv.convexHull(leftEye)
            rightEyeHull = cv.convexHull(rightEye)
            
            cv.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            
            if ear < thresh:
                flag += 1
                print (flag)
                if flag >= frame_check:
                    cv.putText(frame, "****************ALERT!****************", (10, 30),
                               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv.putText(frame, "****************ALERT!****************", (10,325),
                               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
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
                        
        cv.imshow('Raw Webcam Feed', frame)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()