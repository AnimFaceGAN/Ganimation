"""
Rewuire :
    Opencv
    rewuests
    BeautifulSoup
    time
"""

import cv2
import numpy as np
import requests as req
from bs4 import BeautifulSoup as beu
from time import sleep
import os

#画像に顔があるか確認
def get_face(img_path,real_path):
    classifier = cv2.CascadeClassifier(r'D:\CODING\Project\SIGNATE\OpenCV\Casscade model\lbpcascade_animeface.xml')

    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = classifier.detectMultiScale(gray)



    for (x, y, w, h) in faces:

        try:
            face = image[y-20:y + h+10, x-10:x + w+10]
            face = cv2.resize(face, (128, 128))
            cv2.imwrite(real_path, face)
        except cv2.error as e:
            print("CV2 Error were occured!")
            continue
    if len(faces) > 0:
        print(len(faces), "faces were found !!")
        return True




#https://safebooru.org/index.php?page=post&s=list&tags=all&pid=より画像を集める
def colect_girl_images(limit,init=0,num=0):
    url_ori = r"https://safebooru.org/index.php?page=post&s=list&tags=all&pid="
    step = init
    face_id = 0

    url_list=[]
    while 1:
        url=url_ori + str(step)
        print("-----------------------")
        print("Now loading ",url)
        print("-----------------------")
        response = req.get(url)
        soup = beu(response.text, "lxml")

        imgs = soup.find_all("img")
        id = 0

        url_list=list(set(url_list))
        #画像の保存
        for img in imgs:

            if str(img["src"]).find("https://")==-1 :
                img["src"]="https:"+img["src"]
            #print(img["src"])


            if img["src"] in url_list:
                continue

            r = req.get(img["src"])
            sleep(0.001)
            ext=".jpeg"
            if img["src"] in ".jpeg":
                ext=".jpeg"
            elif img["src"] in ".png":
                ext=".png"
            elif img["src"] in ".jpg":
                ext=".jpg"
            with open("../DataSets/Images_for_GAN/Temp_Images"+str(num)+"/temp"+str(id)+ext,"wb") as f:
                f.write(r.content)
            id+=1

        temp_imgs=os.listdir("../DataSets/Images_for_GAN/Temp_Images"+str(num))
        i=0
        #顔の検出
        for temp in temp_imgs:
            img_path="../DataSets/Images_for_GAN/Temp_Images"+str(num)+"/"+temp
            flg=get_face(img_path,"../DataSets/Images_for_GAN/Real_Images/Girl0"+str(num)+"/"+"/girl_"+str(face_id)+".png")

            face_id+=1
            if flg:

                # print(i)
                url_list.append(list(imgs)[i]["src"])


            i+=1

        step-=40

        if len(os.listdir("../DataSets/Images_for_GAN/Real_Images/Girl0"+str(num)+"/"))>=limit:
            break

        print("Collected ",len(os.listdir("../DataSets/Images_for_GAN/Real_Images/Girl0"+str(num)+"/"))," images .\n" )


if __name__=="__main__":
    limit=100000
    colect_girl_images(limit,init=0,num=1)
    #colect_boy_images(limit)







