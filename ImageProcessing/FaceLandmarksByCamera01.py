import dlib
import numpy as np
from imutils import face_utils
import cv2
import csv
import time

# --------------------------------
# 顔ランドマーク検出の前準備
# --------------------------------
# 顔ランドマーク検出ツールの呼び出し
face_detector = dlib.get_frontal_face_detector()
predictor_path = '../DATASET/models/PoseEstimate/shape_predictor_68_face_landmarks.dat'
face_predictor = dlib.shape_predictor(predictor_path)


# --------------------------------
# 特徴量をcsvファイルへ書き込む関数
# --------------------------------
def write_data_to_csv(pts):
    with open('camera_face_features.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerows(pts)


# --------------------------------
# 画像から顔のランドマーク検出する関数
# --------------------------------
def face_landmark_find(img):
    #start = time.time()

    # 顔検出
    img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(img_gry, 1)

    # 検出した全顔に対して処理
    for face in faces:
        # 顔のランドマーク検出
        landmark = face_predictor(img_gry, face)

        # 処理高速化のためランドマーク群をNumPy配列に変換(必須)
        landmark = face_utils.shape_to_np(landmark)

        # 特徴量抽出
        pts1 = landmark[17:22]  # 右眉毛
        pts2 = landmark[22:27]  # 左眉毛
        pts3 = landmark[30]     # 鼻
        pts4 = landmark[36:42]  # 右目
        pts5 = landmark[42:48]  # 左目
        pts6 = landmark[60:68]  # 口

        # 特徴量描画
        cv2.polylines(img, [pts1], False, (0, 0, 0), thickness=2)  # 右眉毛
        cv2.polylines(img, [pts2], False, (0, 0, 0), thickness=2)  # 左眉毛
        cv2.circle(img, tuple(pts3), 3, (0, 255, 0), -1)  # 鼻
        cv2.fillPoly(img, [pts4], (255, 0, 0), lineType=cv2.LINE_8, shift=0)  # 右目
        cv2.fillPoly(img, [pts5], (255, 0, 0), lineType=cv2.LINE_8, shift=0)  # 左目
        cv2.fillPoly(img, [pts6], (0, 0, 255), lineType=cv2.LINE_8, shift=0)  # 口

        # 特徴量をcsvファイルへ書き込む
        write_data_to_csv([np.concatenate([pts1, pts2, [pts3], pts4, pts5, pts6])])

    #elapsed_time = time.time() - start
    #print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    return img


# --------------------------------
# カメラ画像の取得
# --------------------------------
if __name__ == "__main__":
    # カメラの指定(適切な引数を渡す)
    cap = cv2.VideoCapture(0)

    # カメラ画像の表示 ('q'入力で終了)
    while (True):
        ret, img = cap.read()

        # 顔のランドマーク検出(2.の関数呼び出し)
        img = face_landmark_find(img)

        # 結果の表示
        cv2.imshow('img', img)

        # 'q'が入力されるまでループ
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 後処理
    cap.release()
    cv2.destroyAllWindows()
