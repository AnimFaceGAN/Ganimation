import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import  numpy  as np
# from  Animator.Animator import CreateAnimator

class Singleton(object):
    def __new__(cls, *args, **kargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance


class DataBase(Singleton):
    def __name__(self):
        return  "DataBase"

    def __init__(self):
        #resource root path
        self.path_root="./resource/"
        self.AnimeFaces=[]
        self.RealFaces=[]
        self.BaseImageName=os.listdir(self.path_root+"temp/")[0]#"base_image_99"
        self.SettingImage=self.path_root+"temp/"+self.BaseImageName+f"/thumbnail.png"
        #define FileManager to manage folders
        self.fileManager=FileManager(self.path_root)
        self.renderMode="Low" # Rendering mode : "Low" or "High"
        self.stopGenerate=False
        self.finishGenerate=True
        self.generate_log="[00%]  残り時間 000 [sec]"

        #Save Animator for Saving memory
        self.animator=None

        ##########################################################
        #Define Parameters of Screen
        ##########################################################
        
        # 背景画像のパス
        self.output_bg_path = self.path_root+"/Images/save/bg/save1/bg.png"
        self.selected_bg = 1
        #ビデオ画面のupdateオンオフ
        self.playing_video = False
        self.selected_window = "video"
        # アニメ顔画像のパス
        self.output_path = self.path_root+'/Images/output.png'
        self.null_path = self.path_root+'/Images/faceset.png'  # 画像未入力時に表示する
        #Difine Init Params
        self.INITIAL_WIDTH = 1200
        self.INITIAL_HEIGHT = 700
        #カメラデバイスID
        self.CAMERA=0

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
    
    def SetBaseImage(self,imageName):
        self.BaseImageName=imageName
        #self.animator.update_base_image()

class FileManager:
    def __init__(self,path_root):
        # self.root=os.path.dirname(os.path.abspath(__file__))[:-8]+"temp/"
        self.root=path_root+"/temp/"
        self.dir_list=os.listdir(self.root)
        self.folder_temp="base_image_"
        self.image_temp="thumbnail.png"
        self.data_temp="data.pkl"
        self.last_num=max([int(i[i.rfind("_")+1:]) for i in self.dir_list])
    
    def get_image_list(self):
        self.dir_list=os.listdir(self.root)
        return self.dir_list
    
    def get_new_path(self):
        self.last_num=max([int(i[i.rfind("_")+1:]) for i in self.dir_list])+1

        self.dir_list.append(self.folder_temp+str(self.last_num))
        folder_path=self.root+self.folder_temp+str(self.last_num)+"/"
        os.mkdir(folder_path)

        image_path=folder_path+self.image_temp
        data_path=folder_path+self.data_temp
        return image_path,data_path
    
    def get_folder_path(self,folder_name):
        folder_path=self.root+folder_name+"/"
        image_path=folder_path+self.image_temp
        data_path=folder_path+self.data_temp
        return image_path,data_path




