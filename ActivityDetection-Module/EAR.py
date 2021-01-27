from scipy.spatial import distance
import imutils
from imutils import face_utils
import dlib
import cv2 as cv
import numpy as np



def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
	
thresh = 0.25
frame_check = 20		# Time needed to show an alert

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("ActivityDetection-Module/shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code

# Get the left and Right Eyes indices in order to pass calculations
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Take camera Feedback
cap=cv.VideoCapture(0)
flag=0

while True:
	ret, frame=cap.read()
	frame = imutils.resize(frame, width=450)
	gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	subjects = detect(gray, 0)

	for subject in subjects:
		shape = predict(gray, subject)
		shape = face_utils.shape_to_np(shape)			#converting to NumPy Array
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
				#print ("Drowsy")
		else:
			flag = 0
	cv.imshow("Frame", frame)
	key = cv.waitKey(1) & 0xFF
	if key == ord("q"):
		break
cv.destroyAllWindows()
cap.stop()