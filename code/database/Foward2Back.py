from  .BaseDataCenter import BaseDataCenter


class Foward2Back(BaseDataCenter):
    def __init__(self):
        super().__init__()
        pass

    def SetData(self, frame):
        self.database.SetRealFaces(frame)

    def GetData(self):
        return self.database.GetAnimeFaces()

    def SetSettingImage(self,image):
        self.database.SetSettingImage(image)
    
    def SetBaseImage(self, imagefolder):
        pass

    def IsLoadingImage(self):
        return True