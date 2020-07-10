"""
Requires
    imutils
    dlib
    opencv
    scipy
"""

from imutils import face_utils
import imutils
import dlib
import cv2
import os
import time
import  numpy as np
from enum import Enum
from scipy.spatial import distance

class Eye(Enum):
    right=1
    left=2

#DIFINE SOME SETTING
movie_path="..\DATASET/ForFaceDetect/SimpleFaceMovie01.mp4"
save_path="..\DATASET/ForFaceDetect/"
#Define the path of cascade model
cascade_base_path = r"D:\CODING\ANACONDA\envs\Opencv\Library\etc\haarcascades/"
#Define the path of landmarks
model_path="../DATASET/models/PoseEstimate/shape_predictor_68_face_landmarks.dat"

#Prepare for Carcade Model
face_cascade = cv2.CascadeClassifier(os.path.join(cascade_base_path, 'haarcascade_frontalface_alt_tree.xml'))
right_eye_cascade = cv2.CascadeClassifier(os.path.join(cascade_base_path, 'haarcascade_righteye_2splits.xml'))
left_eye_cascade = cv2.CascadeClassifier(os.path.join(cascade_base_path, 'haarcascade_lefteye_2splits.xml'))

#Prepare for Tracking API
tracker = cv2.TrackerMedianFlow_create()
tracker_name = str(tracker).split()[0][1:]

#Prepare for detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_path)

def ChatchFace(frame):
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_points = face_cascade.detectMultiScale(gray)
    return face_points[0]

def ChatchEyes(frame):
    img_gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_right_gray = img_gray[fy:fy + fh, fx:width_center]
    face_left_gray = img_gray[fy:fy + fh, width_center:fx + fw]
    # Detect eyes
    right_eye_points = right_eye_cascade.detectMultiScale(face_right_gray)
    left_eye_points = left_eye_cascade.detectMultiScale(face_left_gray)
    return [right_eye_points,left_eye_points]

def GetFeatures(frame):
    frame = imutils.resize(frame, width=500)
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)
    shape=None

    shape = predictor(gray, rects[0])
    shape = face_utils.shape_to_np(shape)

    return shape

def GetFacePoints(features):
    shape=features

    image_points = np.array([tuple(shape[30]), tuple(shape[8]), tuple(shape[36]), tuple(shape[45]),
                             tuple(shape[48]), tuple(shape[54])], dtype='double')
    return  image_points

def CheckBlink(features,isSide):
    eye=None
    if isSide == Eye.right:
        eye=features[36:42]
    elif isSide ==Eye.left:
        eye=features[42:48]
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    eye_ear = (A + B) / (2.0 * C)
    return round(eye_ear, 3)

def EstimatePose(image_points):
    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ])

    size = frame.shape

    focal_length = size[1]
    center = (size[1] // 2, size[0] // 2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype='double')

    dist_coeffs = np.zeros((4, 1))

    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    (rotation_matrix, jacobian) = cv2.Rodrigues(rotation_vector)
    mat = np.hstack((rotation_matrix, translation_vector))
    # homogeneous transformation matrix (projection matrix)

    (_, _, _, _, _, _, eulerAngles) = cv2.decomposeProjectionMatrix(mat)

    (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                              translation_vector, camera_matrix, dist_coeffs)

    return eulerAngles,nose_end_point2D

def ShowFeatures(image__points,scene):
    for p in image__points:
        cv2.circle(scene, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)


if __name__ =="__main__":
    #Load Movie
    cap=cv2.VideoCapture(movie_path)
    #Get properties of movie
    cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)


    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(os.path.join(save_path, 'detect_face.mp4'), fourcc, fps, (500, 888))


    start = time.time()


    try:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=500)#cv2.resize(frame, (int(cap_width / 4), int(cap_height / 4)))

        #Detect Face
        face=tuple(map(int,ChatchFace(frame)))
        #Init tracker
        tracker.init(frame,face)

        while True:
            if not cap.isOpened():
                break

            ret, frame = cap.read()
            if not ret:
                break
            frame=imutils.resize(frame, width=500)#cv2.resize(frame,(int(cap_width/4), int(cap_height/4)))


            #Track face
            success, roi = tracker.update(frame)
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


            if not success:
                print("Tracking Failed!")
                cv2.putText(frame, "Tracking failed!!", (500, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                continue

            (fx, fy, fw, fh)=tuple(map(int,roi))



            """--- Estimate Face Pose ---"""
            # Get feature
            feature=GetFeatures(frame)
            image_points=GetFacePoints(feature)
            #Estimate
            (yaw,pitch,roll),nose_end_point2D= EstimatePose(image_points)
            #Show Results
            ShowFeatures(image_points,frame)
            #Display Face Direction
            p1 = (int(image_points[0][0]), int(image_points[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

            #Check Blink
            right_blink=CheckBlink(feature,Eye.right)
            left_blink=CheckBlink(feature,Eye.left)

            cv2.line(frame, p1, p2, (255, 0, 0), 2)

            # Face is green
            cv2.rectangle(frame, (fx-2, fy-4), (fx+2 + fw, fy+8 + fh), (0, 255, 0), 2)
            cv2.putText(frame, "face", (fx, fy-2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

            """---  Display Data  ---"""
            cv2.putText(frame, 'yaw : ' + str(int(yaw)), (20, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            cv2.putText(frame, 'pitch : ' + str(int(pitch)), (20, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            cv2.putText(frame, 'roll : ' + str(int(roll)), (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

            cv2.putText(frame, 'Right Blink : {} : {}'.format(str(right_blink),str(right_blink<0.2)) , (20, 55), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            cv2.putText(frame, 'Left Blink : {}  : {}'.format(str(left_blink),str(left_blink<0.2))  , (20, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

            #cv2.putText(frame, '{} FPS'.format(str(round(cap.get(cv2.CAP_PROP_FPS))))  , (50, 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1)

            writer.write(frame)


            # Show result
            if ret:
                cv2.imshow("Frame", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    except cv2.error as e:
        print(e)

    print("ETA {} s".format(time.time() - start))
    writer.release()
    cap.release()
    cv2.destroyAllWindows()
