#YAWN
import cv2
import dlib
import numpy as np
from threading import Thread
import playsound


from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time


PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()
def sound_alarm():
	playsound.playsound("alarm.wav")
	drowsy()
def get_landmarks(im):
    rects = detector(im, 1)
    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im
def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50,53):
        top_lip_pts.append(landmarks[i])
    for i in range(61,64):
        top_lip_pts.append(landmarks[i])
    top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[:,1])
def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65,68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56,59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean[:,1])
def mouth_open(image):
    landmarks = get_landmarks(image)
    if landmarks == "error":
        return image, 0   
    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return image_with_landmarks, lip_distance
def yawn():
    cap = cv2.VideoCapture(0)
    yawns = 0
    yawn_status = False 
    while True:
        ret, frame = cap.read()   
        image_landmarks, lip_distance = mouth_open(frame) 
        prev_yawn_status = yawn_status 
        if lip_distance > 15:
            yawn_status = True
            cv2.putText(frame, "Subject is Yawning", (50,450), 
                        cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)
            output_text = " Yawn Count: " + str(yawns + 1)
            cv2.putText(frame, output_text, (50,50),
                        cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
            if yawns>1:
                cap.release()
                cv2.destroyAllWindows()
                sound_alarm()
                # t = Thread(target=sound_alarm)
                # t.deamon = True
                # t.start()   
        else:
            yawn_status = False 
            
        if prev_yawn_status == True and yawn_status == False:
            yawns += 1
        cv2.imshow('Live Landmarks', image_landmarks )
        cv2.imshow('Yawn Detection', frame ) 
        if cv2.waitKey(1) == 13: 
            break  
   





#DROWSY


def sound_alarm2():
	playsound.playsound("alarm.wav")
def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
def drowsy():
    # cv2.destroyAllWindows()
	ap = argparse.ArgumentParser()
	ap.add_argument("-w", "--webcam", type=int, default=0,
		help="index of webcam on system")
	args = vars(ap.parse_args())
	EYE_AR_THRESH = 0.3
	EYE_AR_CONSEC_FRAMES = 48
	COUNTER = 0
	ALARM_ON = False
	print("[INFO] loading facial landmark predictor...")
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
	(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
	(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
	print("[INFO] starting video stream thread...")
	vs = VideoStream(src=args["webcam"]).start()
	time.sleep(1.0)
	while True:
		frame = vs.read()
		frame = imutils.resize(frame, width=450)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		rects = detector(gray, 0)
		for rect in rects:
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)
			ear = (leftEAR + rightEAR) / 2.0
			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
			if ear < EYE_AR_THRESH:
				COUNTER += 1
				if COUNTER >= EYE_AR_CONSEC_FRAMES:
					if not ALARM_ON:
						ALARM_ON = True
						t = Thread(target=sound_alarm2)
						t.deamon = True
						t.start()
					cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			else:
				COUNTER = 0
				ALARM_ON = False
			cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break
	cv2.destroyAllWindows()
	vs.stop()
yawn()
