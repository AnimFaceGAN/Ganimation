import os
import sys
from tkinter.constants import PROJECTING
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

from Animator.poser.morph_rotate_combine_poser import MorphRotateCombinePoser256Param6
from Animator.puppet.head_pose_solver import HeadPoseSolver
from Animator.poser.poser import Poser
from Animator.puppet.util import compute_left_eye_normalized_ratio, compute_right_eye_normalized_ratio, \
    compute_mouth_normalized_ratio
from Animator.tha.combiner import CombinerSpec
from Animator.tha.face_morpher import FaceMorpherSpec
from Animator.tha.two_algo_face_rotator import TwoAlgoFaceRotatorSpec
from Animator.util import rgba_to_numpy_image, extract_pytorch_image_from_filelike,process_image,show_img

from Animator.GenerateFace import CreateDataGenerator,ImageSaver

from msvcrt import getch

import  database
import time

import pandas as pd
from copy import deepcopy
import asyncio

saver=ImageSaver()

def fire_and_forget(task, *args, **kwargs):
    loop = asyncio.get_event_loop()
    if callable(task):
        return loop.run_in_executor(None, task, *args, **kwargs)
    else:    
        raise TypeError('Task must be a callable')

class Animator:

    def __init__(self,
                 poser: Poser,
                 face_detector,
                 landmark_locator,
                 video_capture,
                 torch_device: torch.device):
        self.database=database.DataLoarder().create_database()
        self.poser = poser
        self.face_detector = face_detector
        self.landmark_locator = landmark_locator
        self.video_capture = video_capture
        self.torch_device = torch_device
        self.head_pose_solver = HeadPoseSolver()

        self.pose_size = len(self.poser.pose_parameters())
        #use to generate image temps
        self.source_image = None
        #use for save image temp
        self.image_temp=None


        self.posed_image = None
        self.current_pose = None
        self.past_pose={"min_index":0,"angle":0}
        self.last_pose = None

        #Create instance of DataGenerator
        self.dataGenerator=CreateDataGenerator(torch_device,poser)

        #self.image_path=os.path.dirname(os.path.abspath(__file__))+"/data/illust/image/face.png"
        self.image_path="./face.png"
        
        #for create source image
        self.source_image = extract_pytorch_image_from_filelike(self.database.SettingImage).to(self.torch_device).unsqueeze(dim=0)
        self.past_image=cv2.imread(self.database.SettingImage)#deepcopy(self.source_image)

        # self.update_base_image()
        self.read_image_temp()

        print("--- Ready OK   ---")

    def update_base_image(self):
        #process_image(self.database.SettingImage)
        print("--- Update Base Image ---")
        self.source_image = extract_pytorch_image_from_filelike(self.database.SettingImage).to(self.torch_device).unsqueeze(dim=0)
        fire_and_forget( self.dataGenerator.create_image)        
    
    def change_base_image(self):
        print("--- Change Base Image ---")
        self.source_image = extract_pytorch_image_from_filelike(self.database.SettingImage).to(self.torch_device).unsqueeze(dim=0)
        self.read_image_temp()

    def update_image(self):
        start=time.time()

        #self.update_base_image()
        #there_is_frame,frame=self.video_capture.read()
        frame = self.database.GetRealFaces()#self.video_capture.read()
        there_is_frame=len(frame)==0
        if there_is_frame:
            print("not detected face 1")
            return self.past_image,False
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.face_detector(rgb_frame)
        euler_angles = None
        face_landmarks = None
        if len(faces) > 0:
            face_rect = faces[0]
            face_landmarks = self.landmark_locator(rgb_frame, face_rect)
            face_box_points, euler_angles = self.head_pose_solver.solve_head_pose(face_landmarks)
            #self.draw_face_landmarks(rgb_frame, face_landmarks)
            # self.draw_face_box(rgb_frame, face_box_points)

        # resized_frame = cv2.flip(cv2.resize(rgb_frame, (192, 256)), 1)
        # pil_image = PIL.Image.fromarray(resized_frame, mode='RGB')

        if euler_angles is not None :
            # print("Estimate Faces")
            self.current_pose = torch.zeros(self.pose_size, device=self.torch_device)
            self.current_pose[0] = max(min(-euler_angles.item(0) / 15.0, 1.0), -1.0)
            self.current_pose[1] = max(min(-euler_angles.item(1) / 15.0, 1.0), -1.0)
            self.current_pose[2] = max(min(euler_angles.item(2) / 15.0, 1.0), -1.0)

            if self.last_pose is None:
                self.last_pose = self.current_pose
            else:
                self.current_pose = self.current_pose * 0.5 + self.last_pose * 0.5
                self.last_pose = self.current_pose

            eye_min_ratio = 0.15
            eye_max_ratio = 0.25
            left_eye_normalized_ratio = compute_left_eye_normalized_ratio(face_landmarks, eye_min_ratio, eye_max_ratio)
            self.current_pose[3] = 1 - left_eye_normalized_ratio
            right_eye_normalized_ratio = compute_right_eye_normalized_ratio(face_landmarks,
                                                                            eye_min_ratio,
                                                                            eye_max_ratio)
            self.current_pose[4] = 1 - right_eye_normalized_ratio

            min_mouth_ratio = 0.02
            max_mouth_ratio = 0.3
            mouth_normalized_ratio = compute_mouth_normalized_ratio(face_landmarks, min_mouth_ratio, max_mouth_ratio)
            self.current_pose[5] = mouth_normalized_ratio

            # if self.current_pose[3]>0.7 or self.current_pose[4]>0.7:
            #     print("[Closing eyes]")
                
            # print(self.current_pose)

            # self.current_pose = self.current_pose.unsqueeze(dim=0)

            # posed_image = self.poser.pose(self.source_image, self.current_pose).detach().cpu()
            numpy_image = self.approx_image()#rgba_to_numpy_image(posed_image[0])
            pil_image = PIL.Image.fromarray(np.uint8(np.rint(numpy_image * 255.0)), mode='RGBA')

            self.database.SetAnimeFaces(numpy_image)

            elapsed_time = time.time() - start
            print(f"\r FPS : {round(1/elapsed_time,2)} [frame/sec]" ,end="")

            self.past_image=numpy_image

            return pil_image,True
        else:
            # print(f"\r not detected face 2",end="")
            self.database.SetAnimeFaces(self.past_image)

            return self.past_image,False

    def approx_image(self):
        start=time.time()
        # print(f"current_pose={self.current_pose}")


        pose_index=["pose_0","pose_1","pose_2","pose_3","pose_4","pose_5"]
        pose_diff=self.image_temp.copy()
        pose_diff[[p+"_diff" for p in pose_index]]=np.abs(pose_diff[pose_index]-self.current_pose.cpu().numpy())

        angles=[]
        angle_1=pose_diff.sort_values(by=["pose_0_diff"], ascending=True)
        angle_1=angle_1[(angle_1["pose_3"]==0) & (angle_1["pose_4"]==0) & (angle_1["pose_5"]==0)]
        angle_2=pose_diff.sort_values(by=["pose_1_diff"], ascending=True)
        angle_2=angle_2[(angle_2["pose_3"]==0) & (angle_2["pose_4"]==0) & (angle_2["pose_5"]==0)]
        angle_3=pose_diff.sort_values(by=["pose_2_diff"], ascending=True)
        angle_3=angle_3[(angle_3["pose_3"]==0) & (angle_3["pose_4"]==0) & (angle_3["pose_5"]==0)]

        angles.append(angle_1.iloc[0].pose_0)
        angles.append(angle_2.iloc[0].pose_1)
        angles.append(angle_3.iloc[0].pose_2)
        min_angle_idx=2#np.argmin(angles)
                
        if np.abs(self.past_pose["angle"])<0.1:
            # pose_diff=pose_diff[pose_diff[pose_index[min_angle_idx]]==angles[min_angle_idx]]
            # pose_diff=pose_diff[pose_diff[pose_index[0]]==angles[0]]
            for i  in range(len(angles)):
                if i == min_angle_idx:
                    pose_diff=pose_diff[pose_diff[pose_index[min_angle_idx]]==angles[min_angle_idx]]
                else:
                    pose_diff=pose_diff[pose_diff[pose_index[i]]==0]
            self.past_pose["min_index"]=min_angle_idx
            self.past_pose["angle"]=angles[min_angle_idx]

        else:
            # pose_diff=pose_diff[pose_diff[pose_index[self.past_pose["min_index"]]]==angles[self.past_pose["min_index"]]]
            for i  in range(len(angles)):
                if i == self.past_pose["min_index"]:
                    pose_diff=pose_diff[pose_diff[pose_index[self.past_pose["min_index"]]]==angles[self.past_pose["min_index"]]]
                else:
                    pose_diff=pose_diff[pose_diff[pose_index[i]]==0]
            
            self.past_pose["angle"]=angles[self.past_pose["min_index"]]

        
        # for i in range(3):
        #     if  i != min_angle_idx:
                #   pose_diff=pose_diff[pose_diff[pose_index[i]]==0]
        #         pass

        # if angles[0]<0.1:
        #     pose_diff=pose_diff[pose_diff[pose_index[min_angle_idx]]==angles[min_angle_idx]]
        # else:
        #     pose_diff=pose_diff[pose_diff[pose_index[0]]==angles[0]]


        parts=[]
        base_img=pose_diff[
                            # (pose_diff.pose_0==angles[0])&(pose_diff.pose_1==angles[1])& (pose_diff.pose_2==angles[2])&
                            (pose_diff["pose_3"]==0) & (pose_diff["pose_4"]==0)&(pose_diff["pose_5"]==0)
                            ].iloc[0].image
        #right eye
        part_1=pose_diff.sort_values(by=["pose_3_diff"], ascending=True)
        part_1=part_1[(part_1["pose_4"]==0) & (part_1["pose_5"]==0)]
        #left eye
        part_2=pose_diff.sort_values(by=["pose_4_diff"], ascending=True)
        part_2=part_2[(part_2["pose_3"]==0) & (part_2["pose_5"]==0)]
        #mouth
        part_3=pose_diff.sort_values(by=["pose_5_diff"], ascending=True)
        part_3=part_3[(part_3["pose_4"]==0) & (part_3["pose_3"]==0)]
        
        parts.append(part_1.iloc[0])
        parts.append(part_2.iloc[0])
        parts.append(part_3.iloc[0])
        # print(f"[ My Choice Poses = angles : {[angles[i] for i in range(3)]} \n, parts :  {[parts[i][pose_index[i+3]] for i in range(3)]} ]")

        # elapsed_time = time.time() - start
        # print("Image Sync ETA : {0} [sec]".format(elapsed_time) )

        return self._sync_image(base_img,parts)

    def _sync_image(self,base_image,parts):
        # sync_image=deepcopy( base_image)
        # c=0
        # for part in parts[::-1]:
        #     gray_part=cv2.cvtColor(part.image, cv2.COLOR_BGR2GRAY)
        #     diff=np.abs((cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)-gray_part))
        #     th=1#np.mean(diff)
        #     sync_image[diff>th]=part.image[diff>th]
        #     c+=1


        sync_image=deepcopy( base_image)
        for part in parts:
            gray_part=cv2.cvtColor(part.image, cv2.COLOR_BGR2GRAY)
            diff=cv2.absdiff(cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY),gray_part)
            th=50#int(np.mean(diff))*3
            # print(pd.DataFrame(pd.Series(diff.ravel()).describe()).transpose())
            ret, img_bin = cv2.threshold(diff, th, 255, 0)
            # contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # 輪郭抽出
            # contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
            contours = cv2.findContours(
                img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

            # 一番面積が大きい輪郭を選択する。
            if len(contours)>0:
                    
                max_cnt = max(contours, key=lambda x: cv2.contourArea(x))
            
                # 黒い画像に一番大きい輪郭だけ塗りつぶして描画する。
                img_bin = np.zeros_like(img_bin)
                cv2.drawContours(img_bin, [max_cnt], -1, color=255, thickness=-1)

                kernel = np.ones((3, 3), np.uint8)
                img_bin = cv2.dilate(img_bin, kernel, iterations = 3)
                # img_bin = cv2.hconcat([img_bin, img_dil])

                # sync_image[out<10]=part.image[out>10]
                contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                for i in range(len(contours)):
                    contour =contours[-1]
                    x, y, width, height = cv2.boundingRect(contour)
                    p=2#min(height,width) #if x>20 and y>20 else 10
                    sync_image[y-p:y+height+p, x-p:x+width+p]=part.image[y-p:y+height+p, x-p:x+width+p]
                

        return sync_image

    def draw_face_box(self, frame, face_box_points):
        line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
                      [4, 5], [5, 6], [6, 7], [7, 4],
                      [0, 4], [1, 5], [2, 6], [3, 7]]
        for start, end in line_pairs:
            cv2.line(frame, face_box_points[start], face_box_points[end], (255, 0, 0), thickness=2)

    def draw_face_landmarks(self, frame, face_landmarks):
        for i in range(68):
            part = face_landmarks.part(i)
            x = part.x
            y = part.y
            frame=cv2.rectangle(frame, (x - 1, y - 1), (x + 1, y + 1), (0, 255, 0), thickness=2)
   
    
    def read_image_temp(self):
        self.image_temp=pd.read_pickle(self.database.fileManager.get_folder_path(self.database.BaseImageName)[1])
        # print(self.database.fileManager.get_folder_path(self.database.BaseImageName))


def CreateAnimator():
    cuda = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    poser = MorphRotateCombinePoser256Param6(
        morph_module_spec=FaceMorpherSpec(),
        morph_module_file_name="data/face_morpher.pt",
        rotate_module_spec=TwoAlgoFaceRotatorSpec(),
        rotate_module_file_name="data/two_algo_face_rotator.pt",
        combine_module_spec=CombinerSpec(),
        combine_module_file_name="data/combiner.pt",
        device=cuda)

    face_detector = dlib.get_frontal_face_detector()
    landmark_locator = dlib.shape_predictor(os.path.dirname(os.path.abspath(__file__))+"\data\shape_predictor_68_face_landmarks.dat")

    video_capture = cv2.VideoCapture(0)

    animator = Animator(poser, face_detector, landmark_locator,video_capture,  cuda)
    return animator

if __name__ == "__main__":
    animator=CreateAnimator()
    print("--- Ready OK !! ---")

    while True:
        key = ord(getch())
        if key==13:
            animator.update_image().show()
        if key==27:
            break

