import cv2 as cv
import mediapipe as mp
import numpy as np
import calculations , draw

# Decalring Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

flag = 0
thresh = 0.2
frame_check = 50

cap = cv.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Process the image
    results = holistic.process(image)
    face_results = results.face_landmarks
    pose_results = results.pose_landmarks

    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    
    if face_results:
        face_landmarks_positions = []
        # assume that only face is present in the image
        for _, data_point in enumerate(face_results.landmark):
            face_landmarks_positions.append([data_point.x, data_point.y, data_point.z]) 
            # saving normalized landmark positions
        face_landmarks_positions = np.array(face_landmarks_positions)
        face_landmarks_positions[:, 0] *= image.shape[1]
        face_landmarks_positions[:, 1] *= image.shape[0]

        # draw face mesh over image
        draw.draw_face(face_results,image,mp_holistic)
        # draw pose
        draw.draw_pose(image,pose_results,mp_holistic)

        EAR = calculations.eye_feature(face_landmarks_positions)

        ## Return boolean if EAR < thresh
        if EAR < thresh:
            flag+=1
            print(flag)
            if flag >= frame_check:
                cv.putText(image, "*ALERT!*", (10, 30),cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv.putText(image, "*ALERT!*", (10,325),cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            flag = 0       
        
    
    landmarks = pose_results.landmark
    # Explain how to extract landmarks Points
    shoulder = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
    elbow = [landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y]
    wrist = [landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].y]

    cv.imshow("Frame", image)
    key = cv.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    
cap.release()
cv.destroyAllWindows()