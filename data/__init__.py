from  Foward2Back import  Foward2Back
from  Back2Forward import Back2Foward
from  DataBase import  DataBase

def GetDataLoarder():
    return  DataLoarder()

class DataLoarder:
    def __init__(self):
        self.database=DataBase()
        self.foward2back=Foward2Back()
        self.back2foward=Back2Foward()
        self.foward2back.initialize(self.database)
        self.back2foward.initialize(self.database)

    def create_foward2back(self):
        return  self.foward2back

    def create_back2foward(self):
        return  self.back2foward

    def create_database(self):
        return  self.database


