import  numpy  as np

class DataBase:
    def __name__(self):
        return  "DataBase"

    def __init__(self):
        self.AnimeFaces=[]
        self.RealFaces=[]
        pass

    def GetAnimeFaces(self):
        return  self.AnimeFaces

    def SetAnimeFaces(self,frame):
        self.AnimeFaces=frame

    def GetRealFaces(self):
        return self.RealFaces

    def SetRealFaces(self,frame):
        self.RealFaces=frame



