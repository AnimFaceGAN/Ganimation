import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import  numpy  as np
from  Animator.Animator import CreateAnimator

class Singleton(object):
    def __new__(cls, *args, **kargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance


class DataBase(Singleton):
    def __name__(self):
        return  "DataBase"

    def __init__(self):
        self.AnimeFaces=[]
        self.RealFaces=[]
        self.SettingImage="C:\Codding\Ganimation\Ganimation\AnimFaceGan\Animator\data\illust\opasity.png"

        #self.animator=CreateAnimator()
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
        #self.animator.update_base_image()





