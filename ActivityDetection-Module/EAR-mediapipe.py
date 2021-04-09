import cv2 as cv
import mediapipe as mp
import numpy as np 
import calculations

# Declaring FaceMesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.3, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils 
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

flag = 0
thresh = 0.2
frame_check = 50

cap = cv.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = face_mesh.process(image)

    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        landmarks_positions = []
        # assume that only face is present in the image
        for _, data_point in enumerate(results.multi_face_landmarks[0].landmark):
            landmarks_positions.append([data_point.x, data_point.y, data_point.z]) 
            # saving normalized landmark positions
        landmarks_positions = np.array(landmarks_positions)
        landmarks_positions[:, 0] *= image.shape[1]
        landmarks_positions[:, 1] *= image.shape[0]

        # draw face mesh over image
        for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

        ear = calculations.eye_feature(landmarks_positions)

        if ear < thresh:
            flag+=1
            print(flag)
            if flag >= frame_check:
                cv.putText(image, "*ALERT!*", (10, 30),cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv.putText(image, "*ALERT!*", (10,325),cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            flag = 0

    cv.imshow("Frame", image)
    key = cv.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv.destroyAllWindows()
cap.stop()