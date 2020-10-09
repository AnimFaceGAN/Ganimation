#Requires : https://github.com/kanosawa/anime_face_landmark_detection
import cv2
import  os
import sys
import dlib
import numpy as np
from imutils import face_utils
import csv
import torch
from torchvision import transforms
import cv2
from PIL import Image, ImageDraw
from CFA import CFA
from tqdm import tqdm
#import animeface

# 顔ランドマーク検出ツールの呼び出し
face_detector = dlib.get_frontal_face_detector()
predictor_path = "../DataSets/models/dlib/shape_predictor_68_face_landmarks.dat"
face_predictor = dlib.shape_predictor(predictor_path)

# Images Property
img_height=280
img_width=280
rotate_max=90
rotate_min=-90
resize_max=1.4
resize_min=0.6
shift_max=(50,50)
shift_min=(-50,-50)
max_frame=30

test_per=0.2

args = sys.argv
try:
    root_folder=args[1]
except IndexError:
    root_folder="C:\Codding\Ganimation\DataSets\Images_for_GAN\Real_Images/Girl01"

try:
    save_folder=args[2]
except IndexError:
    save_folder=r"C:\Codding\Ganimation\few-shot-vid2vid-master\datasets"

def write_data_to_csv(path,pts):
    with open(path, 'a') as f:
        writer = csv.writer(f)
        writer.writerows(pts)

def GetCheckPoint(img):
    # param
    num_landmark = 24
    img_width = 128
    checkpoint_name = '../anime_face_landmark_detection-master/checkpoint_landmark_191116.pth.tar'

    # detector
    face_detector = cv2.CascadeClassifier('../anime_face_landmark_detection-master/lbpcascade_animeface.xml')
    landmark_detector = CFA(output_channel_num=num_landmark + 1, checkpoint_name=checkpoint_name).cuda()

    # transform
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    train_transform = [transforms.ToTensor(), normalize]
    train_transform = transforms.Compose(train_transform)

    # input image & detect face
    #faces = face_detector.detectMultiScale(img)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))



    # transform image
    #img_tmp = img.crop((x, y, x + w, y + h))
    img_tmp = img.resize((img_width, img_width), Image.BICUBIC)
    img_tmp = train_transform(img_tmp)
    img_tmp = img_tmp.unsqueeze(0).cuda()

    # estimate heatmap
    heatmaps = landmark_detector(img_tmp)
    heatmaps = heatmaps[-1].cpu().detach().numpy()[0]

    landmarks=[]
    # calculate landmark position
    for i in range(num_landmark):
        heatmaps_tmp = cv2.resize(heatmaps[i], (img_width, img_width), interpolation=cv2.INTER_CUBIC)
        landmark = np.unravel_index(np.argmax(heatmaps_tmp), heatmaps_tmp.shape)
        landmark_y = landmark[1] * (280//128)#(img_height//np.shape(heatmaps_tmp)[1])
        landmark_x = landmark[0] * (280//128)#(img_width//np.shape(heatmaps_tmp)[0])
        landmark=[landmark_x,landmark_y]
        landmarks.append(landmark)

    return  landmarks


def extra_point(checkpoints):
    points=[]
    for i in range(np.shape(checkpoints)[0]):
        for j in range(np.shape(checkpoints)[1]):
            if checkpoints[i][j]>0:
                points.append([i,j])
    l_point=len(points)
    n_point=np.arange(0,l_point)
    if l_point>24:
        k_points=np.random.choice(n_point,24,replace=False)
        #points=points[n_point]


    return  points

def augmantation(img,magn,ang,shift):
   #n_img = cv2.resize(img, (int(img_width * magn), int(img_height * magn))
    center = (int(img_width / 2), int(img_height / 2))
    # スケールを指定
    scale =  magn
    # getRotationMatrix2D関数を使用
    trans = cv2.getRotationMatrix2D(center, ang, scale)
    # アフィン変換
    image2 = cv2.warpAffine(img, trans, (img_width, img_height))

    M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    image2=cv2.warpAffine(image2,M,(img_width,img_height))

    return image2

def main_process():
    make_dir()
    list_dir=os.listdir(root_folder)

    test_num=int(len(list_dir)*test_per)

    counter=0
    for img_path in tqdm(list_dir):
        img_dir=save_folder+"/"+"AnimeFace"+"/"+"train_images/"+"face"+str(counter)
        csv_dir=save_folder+"/"+"AnimeFace"+"/"+"train_keypoints/"+"label"+str(counter)
        if counter>= test_num:
            img_dir = save_folder + "/" + "AnimeFace" + "/" + "test_images/" + "face" + str(counter-test_num)
            csv_dir = save_folder + "/" + "AnimeFace" + "/" + "test_keypoints/" + "label" + str(counter-test_num)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(csv_dir, exist_ok=True)

        img_path=root_folder+"/"+img_path
        img=cv2.imread(img_path)
        img=cv2.resize(img,(img_width,img_height))

        white_img=np.zeros((img_height, img_width))
        white_img=cv2.resize(white_img,(img_width,img_height))

        checkpoints=GetCheckPoint(img)

        color =255
        for point in checkpoints:
            try :
                white_img[point[0]][point[1]]=color

            except IndexError:
                pass

        #Reshape Property
        max_resize=np.random.uniform(resize_min,resize_max)
        max_rotation=np.random.uniform(rotate_min,rotate_max)
        max_shift=[np.random.uniform(shift_min[0],shift_max[0]),np.random.uniform(shift_min[1],shift_max[1])]
        resize_step=(max_resize-1)/max_frame
        rotate_step=max_rotation/max_frame
        shift_step=np.array(max_shift)/max_frame



        for i in range(max_frame):
            magn=i*resize_step+1
            ang=i*rotate_step
            shift=i*shift_step

            n_img=augmantation(img,magn,ang,shift)
            n_white_img=augmantation(white_img,magn,ang,shift)



            #save images
            cv2.imwrite(img_dir+"/"+"img"+str(i)+".png",n_img)
            #save labels
            extra_checkpoint=extra_point(n_white_img)
            """
            show_img(n_img)
            show_img(n_white_img)
            k_white_img = np.zeros((img_height, img_width))
            for point in extra_checkpoint:
                try:
                    k_white_img[point[0]][point[1]] = color

                except IndexError:
                    pass
            show_img(k_white_img,title="white")
            #"""


            write_data_to_csv(csv_dir+"/"+"label"+str(i)+".txt",extra_checkpoint)


        counter+=1


def make_dir():
    os.makedirs(save_folder + "/" + "AnimeFace", exist_ok=True)
    os.makedirs(save_folder + "/" + "AnimeFace/" + "train_images", exist_ok=True)
    os.makedirs(save_folder + "/" + "AnimeFace/" + "train_keypoints", exist_ok=True)
    os.makedirs(save_folder + "/" + "AnimeFace/" + "test_images", exist_ok=True)
    os.makedirs(save_folder + "/" + "AnimeFace/" + "test_keypoints", exist_ok=True)

def show_img(img,title=""):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__=="__main__":
    main_process()



