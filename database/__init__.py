import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from  .Foward2Back import  Foward2Back
from  .Back2Forward import Back2Foward
from  .DataBase import  DataBase

def GetDataLoarder():
    return  DataLoarder()

class Singleton(object):
    def __new__(cls, *args, **kargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance

class DataLoarder(Singleton):
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




