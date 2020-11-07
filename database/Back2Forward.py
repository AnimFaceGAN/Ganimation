from  .BaseDataCenter import BaseDataCenter

class Back2Foward(BaseDataCenter):
    def __init__(self):
        super().__init__()
        pass

    def SetData(self,frame):
        self.database.SetAnimeFaces(frame)

    def GetData(self):
        return  self.database.GetRealFaces()



