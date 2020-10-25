import cv2

# --------------------------------
# アニメ顔検出の前準備
# --------------------------------
cascade_path = '../models/lbpcascade_animeface.xml'
input_path = '../images/input.png'
output_path = '../images/output.png'
failure_path = '../images/failure.png'
resize_lenght = 280

# --------------------------------
# アニメ画像から顔を検出する関数　引き数：対象の画像
# --------------------------------
def anime_face_detect(img):
    print('start detecting anime face')

    output_img = None

    # 顔検出
    cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)


    faces = cascade.detectMultiScale(gray,
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(25, 25))

    # 顔の切り抜き、リサイズ
    max_w = 0 # wの最大値を格納する
    for (x, y, w, h) in faces:
        # 顔が複数検出された場合、サイズが最も大きいものを出力
        if(max_w < w):
            output_img = img[y:y + h, x:x + w]
            output_img = cv2.resize(output_img, (resize_lenght, resize_lenght))
            max_w = w

    if output_img is not None:
        cv2.imwrite(output_path, output_img)
        print('->success')
        return True
    else:
        fail_img = cv2.imread(failure_path, cv2.IMREAD_COLOR)
        cv2.imwrite(output_path, fail_img)
        print('->fail')
        return False

# --------------------------------
# アニメ顔画像の入力
# --------------------------------
if __name__ == "__main__":

    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    anime_face_detect(img)
