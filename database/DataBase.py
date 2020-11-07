import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import  numpy  as np
from  Animator.Animator import CreateAnimator

class DataBase:
    def __name__(self):
        return  "DataBase"

    def __init__(self):
        self.AnimeFaces=[]
        self.RealFaces=[]
        self.SettingImage=r"C:\Codding\Ganimation\Ganimation\AnimFaceGan\Animator\data\illust\waifu_02_256.png"
        self.animator=CreateAnimator()
        pass

    def GetAnimeFaces(self):
        return  self.AnimeFaces

    def SetAnimeFaces(self,frame):
        self.AnimeFaces=frame

    def GetRealFaces(self):
        return self.RealFaces

    def SetRealFaces(self,frame):
        self.RealFaces=frame

    def SetSettingImage(self,image):
        self.SettingImage=image

    def update(self):
        self.animator.update_image()



