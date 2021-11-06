#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
import os
import re
from natsort import natsorted
import pickle
import shutil

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import pathlib
import imghdr
from anime_face_landmark.AnimeFaceDetect import anime_face_detect
from  database import  *
import cv2
import os.path
import  numpy as np
import time

import numpy as np
from kivy.app import App
from kivy.lang import Builder
from kivy.properties import StringProperty, ObjectProperty
from kivy.core.window import Window
from kivy.uix.screenmanager import (ScreenManager, Screen, NoTransition, SlideTransition, CardTransition, SwapTransition, FadeTransition, WipeTransition, FallOutTransition, RiseInTransition)

from kivy.uix.image import Image as KImage
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.uix.screenmanager import (ScreenManager, Screen, NoTransition, SlideTransition, CardTransition,
                                    SwapTransition, FadeTransition, WipeTransition, FallOutTransition, RiseInTransition)
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button

#for pupup
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout

# 日本語フォント表示対応
from kivy.core.text import LabelBase, DEFAULT_FONT
from kivy.resources import resource_add_path

#from  Animator.Animator import CreateAnimator

from kivy.config import Config
Config.set('graphics', 'resizable', False)

import subprocess
from functools import partial

from  Animator.Animator import CreateAnimator
from  copy import deepcopy
from PIL import Image

import asyncio
def fire_and_forget(task, *args, **kwargs):
    loop = asyncio.get_event_loop()
    if callable(task):
        return loop.run_in_executor(None, task, *args, **kwargs)
    else:
        raise TypeError('Task must be a callable')

#for using virtual camera
import pyvirtualcam

# DataBaseのインスタンス化
# DataLoarderクラスのインスタンス化
loarder=DataLoarder()
#それぞれのインスタンスを読み込む
f2b=loarder.create_foward2back()
b2f=loarder.create_back2foward()
DB=loarder.create_database()

#Define root
path_root=DB.path_root
# フォント読み込み OS関係無し
resource_add_path(path_root+'/Font')
LabelBase.register(DEFAULT_FONT, 'ipaexg.ttf')
# Kivyファイルの読み込み
Builder.load_file(path_root+'/component/OtherSettingsScreen.kv', encoding="utf-8")
#Difine Init Params
# INITIAL_WIDTH = DB.INITIAL_WIDTH
# INITIAL_HEIGHT = DB.INITIAL_HEIGHT
# Window.size = (INITIAL_WIDTH, INITIAL_HEIGHT)


# その他設定画面
class OtherSettingsScreen(Screen):
    popup_close = ObjectProperty(None)
    to_settings0 = ObjectProperty(None)
    to_settings1 = ObjectProperty(None)
    to_settings2 = ObjectProperty(None)

    def __init__(self, **kwargs):
        # super().__init__(**kwargs)
        super(OtherSettingsScreen, self).__init__(**kwargs)
        self.ids.camera_id.text = "Camera Device ID "+str(DB.CAMERA)
        self.ids.render_mode.text = DB.renderMode
        self.ids.camera_id.text = str(DB.cameras[0])#str(CAMERA)

    def ChangeCamera(self, num):
        if (-1 < DB.CAMERA + num) and (DB.CAMERA + num < 10):
            CAMERA = DB.CAMERA + num
            if len(DB.cameras)<CAMERA+1:
                return
            self.ids.camera_id.text = str(DB.cameras[CAMERA])#str(CAMERA)
            data = {"Camera":CAMERA, "RenderingMode":DB.renderMode}
            with open(DB.path_root+"save/setting_data.pickle", mode="wb") as f:
                pickle.dump(data, f)

    def ChangeRenderingMode(self):
        if DB.renderMode == "Low":
            DB.renderMode = "High"
            self.ids.render_mode.text = "High"

        elif DB.renderMode == "High":
            DB.renderMode = "Low"
            self.ids.render_mode.text = "Low"

        data = {"Camera":DB.CAMERA, "RenderingMode":DB.renderMode}
        with open(DB.path_root+"save/setting_data.pickle", mode="wb") as f:
            pickle.dump(data, f)

    def ChangeVirtualCamera(self):
        if DB.virtual_camera :
            DB.virtual_camera = False
            self.ids.virtual_camera.text = "OFF"

        elif not DB.virtual_camera:
            DB.virtual_camera = True
            self.ids.virtual_camera.text = "ON"

        data = {"Camera":DB.CAMERA, "RenderingMode":DB.renderMode}
        with open(DB.path_root+"save/setting_data.pickle", mode="wb") as f:
            pickle.dump(data, f)
