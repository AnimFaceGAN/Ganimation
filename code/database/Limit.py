import sys
import os
import numpy as np
from datetime import datetime 
import time
import pickle
# from Crypto.Cipher import AES
# from Crypto.Random import get_random_bytes

class LimitApp:
    def __init__(self,path_root,limit=True):
        self.LIMIT=limit

        #Parameters
        self.LIMIT_TIME=50 *60 #Minutue
        
        #Condirions
        self.date=datetime.now()
        self.timestump=time.time()
        self.rest_time=self.LIMIT_TIME

        #パスを設定する
        self.path_root=path_root
        #暗号化するか
        self.encypt=False
        self._load()



    def check(self):
        if not self.LIMIT:
            return False
        
        now=datetime.now()
        now_time=time.time()
        # if (self.date.year-now.year)<=0 and (self.date.month-now.month)<=0 and  (self.date.day-now.day)<=0:
        if (now-self.date).days<=0:
            self.rest_time=self.rest_time-(now_time-self.timestump)
        else:
            self.rest_time=self.LIMIT_TIME
        self.date=datetime.now()
        self.timestump=now_time

        if not self._interval(step=60):
            self._save()
            #return False
        if self.rest_time<0:
            return True
        else:
            return False
    def _interval(self,step=60):
        if (time.time() - self.timestump)>step:
            return True
        return False
        
    def _load(self):
        if not os.path.exists(self.path_root+"save/passdata.pickle"):
            self.date=datetime.now()
            self.rest_time=self.LIMIT_TIME
            return
        with open(self.path_root+"save/passdata.pickle", mode="rb") as f:
            data = pickle.load(f)
        self._decode(data)
        return
    
    def _save(self):
        if not self.encypt:
            data={"date":self.date,"date_tag":None,"rest_time":self.rest_time,"rest_time_tag":None,"nonce":None,"key":None}
            with open(self.path_root+"save/passdata.pickle", mode="wb") as f:
                pickle.dump(data,f)
            return

        # key = get_random_bytes(16)
        # cipher = AES.new(key, AES.MODE_EAX)
        # date, date_tag = cipher.encrypt_and_digest(bytes( self.date.strftime('%Y-%m-%d %H:%M:%S'), encoding='utf8'))
        # cipher = AES.new(key, AES.MODE_EAX)
        # rest_time, rest_time_tag = cipher.encrypt_and_digest(bytes( str(self.rest_time), encoding='utf8'))
        # data={"date":date,"date_tag":date_tag,"rest_time":rest_time,"rest_time_tag":rest_time_tag,"nonce":cipher.nonce,"key":key}
        # with open(self.path_root+"save/passdata.pickle", mode="wb") as f:
        #     pickle.dump(data,f)
        # return

    def _decode(self,data):
        if not self.encypt:
            self.date=data["date"]
            self.rest_time=data["rest_time"]
            return

        # cipher_dec = AES.new(data["key"], AES.MODE_EAX, data["nonce"])
        # date=str( cipher_dec.decrypt_and_verify(data["date"], data["date_tag"]))
        # self.date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        # self.rest_time=float( cipher_dec.decrypt_and_verify(data["rest_time"], data["rest_time_tag"]))
        # return

