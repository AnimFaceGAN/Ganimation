import dlib
from imutils import face_utils
import cv2

# --------------------------------
# 1.顔ランドマーク検出の前準備
# --------------------------------
# 顔ランドマーク検出ツールの呼び出し
face_detector = dlib.get_frontal_face_detector()
predictor_path = '../DATASET/models/PoseEstimate/shape_predictor_68_face_landmarks.dat'
face_predictor = dlib.shape_predictor(predictor_path)


# --------------------------------
# 2.画像から顔のランドマーク検出する関数
# --------------------------------
def face_landmark_find(img):
    # 顔検出
    img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(img_gry, 1)
    #print(faces)

    # 検出した全顔に対して処理
    for face in faces:
        # 顔のランドマーク検出
        landmark = face_predictor(img_gry, face)

        # 処理高速化のためランドマーク群をNumPy配列に変換(必須)
        landmark = face_utils.shape_to_np(landmark)

        # ランドマーク描画

        #右眉毛
        pts = landmark[17:22]
        cv2.polylines(img, [pts], False, (0, 0, 0), thickness=2)

        #左眉毛
        pts = landmark[22:27]
        cv2.polylines(img, [pts], False, (0, 0, 0), thickness=2)

        #鼻
        (x, y) = landmark[30]
        cv2.circle(img, (x, y), 3, (0, 255, 0), -1)

        #右目
        pts = landmark[36:42]
        cv2.fillPoly(img, [pts], (255, 0, 0), lineType=cv2.LINE_8, shift=0)

        #左目
        pts = landmark[42:48]
        cv2.fillPoly(img, [pts], (255, 0, 0), lineType=cv2.LINE_8, shift=0)

        #口
        pts = landmark[60:68]
        cv2.fillPoly(img, [pts], (0, 0, 255), lineType=cv2.LINE_8, shift=0)

        """"
        for (x, y) in landmark:
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
        """

    return img


# --------------------------------
# 3.カメラ画像の取得
# --------------------------------
if __name__ == "__main__":
    # カメラの指定(適切な引数を渡す)
    cap = cv2.VideoCapture(0)

    # カメラ画像の表示 ('q'入力で終了)
    while(True):
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