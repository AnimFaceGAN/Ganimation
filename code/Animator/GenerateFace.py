
#Tips
#なんか顔の角度で端から二番目あたりが正面に向くようになってる要検討

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from tkinter import Tk, Frame, LEFT, Label, BOTH, GROOVE, Button, filedialog, PhotoImage, messagebox

import numpy as np
import PIL.Image
import PIL.ImageTk
import cv2
import dlib
import torch
import pandas as pd

from Animator.poser.morph_rotate_combine_poser import MorphRotateCombinePoser256Param6
from Animator.puppet.head_pose_solver import HeadPoseSolver
from Animator.poser.poser import Poser
from Animator.puppet.util import compute_left_eye_normalized_ratio, compute_right_eye_normalized_ratio, \
    compute_mouth_normalized_ratio
from Animator.tha.combiner import CombinerSpec
from Animator.tha.face_morpher import FaceMorpherSpec
from Animator.tha.two_algo_face_rotator import TwoAlgoFaceRotatorSpec
from Animator.util import rgba_to_numpy_image, extract_pytorch_image_from_filelike,process_image,show_img

from msvcrt import getch

import  database

import asyncio

# from poser.morph_rotate_combine_poser import MorphRotateCombinePoser256Param6
# from puppet.head_pose_solver import HeadPoseSolver
# from poser.poser import Poser
# from puppet.util import compute_left_eye_normalized_ratio, compute_right_eye_normalized_ratio, \
#     compute_mouth_normalized_ratio
# from tha.combiner import CombinerSpec
# from tha.face_morpher import FaceMorpherSpec
# from tha.two_algo_face_rotator import TwoAlgoFaceRotatorSpec
# from util import rgba_to_numpy_image, extract_pytorch_image_from_filelike,process_image,show_img

# from msvcrt import getch

# import  database

import time
#from tqdm import tqdm


class FPS:
    def __init__(self):
        import time
        self.s1=time.time()
        self.i=0
        self.c=0

    def start(self):
        self.s1=time.time()
        self.i=0
        self.c=0

    def stop(self):
        c= round(time.time()-self.s1,4)+1e-8
        print(f"second : {round(c,4)} , diff {round(c-self.c,5)}, fps : {round(1/c,4)}  ---{self.i}")
        self.i+=1
        self.c=c
        return c
    def end(self):
        print("-----------------------------------------------------------------------------/n")

fps=FPS()


class ImageSaver:
    def __init__(self,root="/temp/"):
        self.image_idx=0
        self.root=os.path.dirname(os.path.abspath(__file__))[:-9]+root
        self.temp="anime_face_"
        self.ext=".png"

    def save(self,image):
        path=self.root+self.temp+str(self.image_idx)+self.ext
        cv2.imwrite(path, image)
        self.image_idx+=1

class ImageManager:
    def __init__(self,data,root="/temp/"):
        self.database=data

        self.root=os.path.dirname(os.path.abspath(__file__))[:-9]+root
        self.temp="anime_faces"
        self.ext=".pkl"

class DataGenerator:

    def __init__(self,
                 poser: Poser,
                 torch_device: torch.device):
        self.database=database.DataLoarder().create_database()
        self.poser = poser
        # self.video_capture = video_capture
        self.torch_device = torch_device
        self.head_pose_solver = HeadPoseSolver()

        self.pose_size = len(self.poser.pose_parameters())
        self.source_image = None
        self.posed_image = None
        self.current_pose = None
        self.last_pose = None

        #Create Anime Images database
        self.images_temp=pd.DataFrame([],columns=["pose_0","pose_1","pose_2","pose_3","pose_4","pose_5","image"])
        self.set_images_temp()

        #self.image_path=os.path.dirname(os.path.abspath(__file__))+"/data/illust/image/face.png"
        self.image_path="./face.png"

        self.update_base_image()


    def update_base_image(self):
        #process_image(self.database.SettingImage)
        self.source_image = extract_pytorch_image_from_filelike(self.database.SettingImage).to(self.torch_device).unsqueeze(dim=0)

    def set_images_temp(self):
        # start=time.time()
        # frame = self.database.GetRealFaces()#self.video_capture.read()
        self.update_base_image()
        # imageSaver=ImageSaver()
        self.current_pose = torch.zeros(self.pose_size, device=self.torch_device)
        
        # print("Start Generate Images")
        #Reset image temp
        self.images_temp=pd.DataFrame([],columns=["pose_0","pose_1","pose_2","pose_3","pose_4","pose_5","image"])
        
        euler_angles_list=[ -1 , -0.8 , -0.6 , -0.4 , -0.2 , 0 , 0.2 , 0.4 , 0.6 , 0.8, 1]#np.round(np.arange(-1,1,0.2),1)#np.linspace(-1,1,10)
        eye_ratio_list=np.linspace(0,1,5)
        mouth_ratio_list=[0,0.5,1]#np.linspace(0,1,5)

        #############################################################################
        # USE ONLY 1 AXIS TO MOVE FACE
        #############################################################################
        for eul_r in euler_angles_list:
            for euler_idx in [2]:
                _l=np.array([0,1,2])
                _l=_l[~(_l==euler_idx)]
                self.current_pose[euler_idx]=eul_r
                self.current_pose[_l[0]]=0
                self.current_pose[_l[1]]=0
                
                which__parts={
                    "r_eye":eye_ratio_list,
                    "l_eye":eye_ratio_list,
                    "mouth":mouth_ratio_list
                    }
                
                pose_idx=["pose_0","pose_1","pose_2","pose_3","pose_4","pose_5"]

                for part_name,c_pose  in which__parts.items():
                    if part_name=="r_eye":
                        for i in c_pose:
                            self.current_pose[3]=i
                            self.current_pose[4]=0
                            self.current_pose[5]=0
                            _temp=pd.DataFrame({pose_idx[i] : [float(self.current_pose[i])]  for i in range(len(pose_idx))})
                            self.images_temp=self.images_temp.append(pd.DataFrame(_temp))

                            # numpy_image=self.create_anime_from_pose()
                            # imageSaver.save(numpy_image)
                
                    if part_name=="l_eye":
                        for i in c_pose:
                            self.current_pose[3]=0
                            self.current_pose[4]=i
                            self.current_pose[5]=0
                            _temp=pd.DataFrame({pose_idx[i] : [float(self.current_pose[i])]  for i in range(len(pose_idx))})
                            self.images_temp=self.images_temp.append(pd.DataFrame(_temp))

                            # numpy_image=self.create_anime_from_pose()
                            # imageSaver.save(numpy_image)

                    if part_name=="mouth":
                        for i in c_pose:
                            self.current_pose[3]=0
                            self.current_pose[4]=0
                            self.current_pose[5]=i
                            _temp=pd.DataFrame({pose_idx[i] : [float(self.current_pose[i])]  for i in range(len(pose_idx))})
                            self.images_temp=self.images_temp.append(pd.DataFrame(_temp))

                            # numpy_image=self.create_anime_from_pose()
                            # imageSaver.save(numpy_image)
        self.images_temp.reset_index(inplace=True)
        # print(self.images_temp)

    def create_image(self):
        self.update_base_image()
        self.set_images_temp()
        start=time.time()

        frame = self.database.GetRealFaces()#self.video_capture.read()
        there_is_frame=len(frame)==0
        euler_angles = None
        face_landmarks = None
        
        self.current_pose = torch.zeros(self.pose_size, device=self.torch_device)
        
        print("Start Generate Images")

        euler_angles_list=np.linspace(-1,1,10)
        eye_ratio_list=np.linspace(0,1,5)
        mouth_ratio_list=np.linspace(0,1,5)

        pose_idx=["pose_0","pose_1","pose_2","pose_3","pose_4","pose_5"]

        print("--- Update Base Image ---")

        for i in range(len(self.images_temp)):
            if self.database.stopGenerate:
                break
            print(f"\r[{i}/{len(self.images_temp)}: Created Images | time : {round(time.time() - start,3)}[sec]]", end='')
            
            #current_poseに値を突っ込む
            for j in range(len(self.current_pose)):
                self.current_pose[j]=self.images_temp.iloc[i][pose_idx[j]]#.values[j]
            numpy_image=self.create_anime_from_pose()
            # print(self.images_temp.loc[[self.images_temp.index[i]],"image"])
            self.images_temp.loc[[self.images_temp.index[i]],"image"]=[numpy_image]
            _eta=time.time() - start
            self.database.generate_log=f"[{int(100*i/len(self.images_temp))}%]  残り時間 : {round(_eta*(len(self.images_temp)/(i+1)))}[sec]"

        self.database.generate_log="[00%]  残り時間 000 [sec]"
            # imageSaver.save(numpy_image)
        if self.database.stopGenerate:
            print("Canceled Prerendering")
            self.database.finishGenerate=True
            return
        
        #save image and dataframe
        self.save_data()
        elapsed_time = time.time() - start
        print(f"\r Total time : {elapsed_time} [sec]" ,end="")

        print("Finish the works!")
        self.database.finishGenerate=True

        return True
    
    def save_data(self):
        image_path,data_path=self.database.fileManager.get_new_path()
        cv2.imwrite(image_path, cv2.imread(self.database.SettingImage))
        # cv2.imwrite(image_path, self.source_image.cpu().numpy())
        self.images_temp.to_pickle(data_path)

        #Reset Image Temp to reduce memory
        self.images_temp=pd.DataFrame([],columns=["pose_0","pose_1","pose_2","pose_3","pose_4","pose_5","image"])


    def create_anime_from_pose(self):
        current_pose = self.current_pose.unsqueeze(dim=0)
        posed_image = self.poser.pose(self.source_image, current_pose).detach().cpu()
        numpy_image = rgba_to_numpy_image(posed_image[0])

        return np.array(numpy_image*255,dtype=np.uint8)


def CreateDataGenerator(cuda,poser):

    animator = DataGenerator(poser,  cuda)
    return animator



