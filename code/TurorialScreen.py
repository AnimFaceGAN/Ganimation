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
Builder.load_file(path_root+'/component/TutorialScreen.kv', encoding="utf-8")
#Difine Init Params
INITIAL_WIDTH = DB.INITIAL_WIDTH
INITIAL_HEIGHT = DB.INITIAL_HEIGHT
Window.size = (INITIAL_WIDTH, INITIAL_HEIGHT)

# チュートリアル画面
class TutorialScreen(Screen):
    popup_close = ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def ChangeStep(self, num):
        self.ChangeOffColor()
        if num == 1:
            self.ids.b1.background_color=(0.0, 0.6, 0.7, 1)
            self.ids.display_image.source = path_root+'/Images/UI_images/Tutorial/Video.png'
            self.ids.display_text.text = '「Ganimation」はたった一枚の顔イラストからアバターを生成できます。\n機械学習の応用技術をGan使用することで実現しています。\n作成したアバターでZoomやDiscordなどに参加できます。'
        elif num == 2:
            self.ids.b2.background_color=(0.0, 0.6, 0.7, 1)
            self.ids.display_image.source = path_root+'/Images/UI_images/Tutorial/Prerendering.png'
            self.ids.display_text.text = 'プリレンダリングでアバターを生成しましょう。'
        elif num == 3:
            self.ids.b3.background_color=(0.0, 0.6, 0.7, 1)
            self.ids.display_image.source = path_root+'/Images/UI_images/Tutorial/AvatarSelect.png'
            self.ids.display_text.text = 'アバターを選択しましょう。'
        elif num == 4:
            self.ids.b4.background_color=(0.0, 0.6, 0.7, 1)
            self.ids.display_image.source = path_root+'/Images/UI_images/Tutorial/BG.png'
            self.ids.display_text.text = '背景も設定しましょう。'
        elif num == 5:
            self.ids.b5.background_color=(0.0, 0.6, 0.7, 1)
            self.ids.display_image.source = path_root+'/Images/UI_images/Tutorial/Camera.png'
            self.ids.display_text.text = '使用するカメラを選択しましょう。'
        elif num == 6:
            self.ids.b6.background_color=(0.0, 0.6, 0.7, 1)
            self.ids.display_image.source = path_root+'/Images/UI_images/Tutorial/Video.png'
            self.ids.display_text.text = '準備が出来たらビデオ画面に戻ります。\nDiscordなどのビデオアプリに接続しアバターを使ってみましょう。'



    def ChangeOffColor(self):
        self.ids.b1.background_color=(0.3, 0.3, 0.3, 1.0)
        self.ids.b2.background_color=(0.3, 0.3, 0.3, 1.0)
        self.ids.b3.background_color=(0.3, 0.3, 0.3, 1.0)
        self.ids.b4.background_color=(0.3, 0.3, 0.3, 1.0)
        self.ids.b5.background_color=(0.3, 0.3, 0.3, 1.0)
        self.ids.b6.background_color=(0.3, 0.3, 0.3, 1.0)



