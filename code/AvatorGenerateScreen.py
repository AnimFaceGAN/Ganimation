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
#Define Animator
if DB.animator==None:
    DB.animator=CreateAnimator()
# フォント読み込み OS関係無し
resource_add_path(path_root+'/Font')
LabelBase.register(DEFAULT_FONT, 'ipaexg.ttf')
# Kivyファイルの読み込み
Builder.load_file(path_root+'/component/AvatarGenerateScreen.kv', encoding="utf-8")
#Difine Init Params
# INITIAL_WIDTH = DB.INITIAL_WIDTH
# INITIAL_HEIGHT = DB.INITIAL_HEIGHT
# Window.size = (INITIAL_WIDTH, INITIAL_HEIGHT)

#TEST REDERING MODE
# DB.renderMode="Low" # "Low"  or  "High"






# アバター生成画面
class AvatarGenerateScreen(Screen):
    popup_close = ObjectProperty(None)
    to_settings1 = ObjectProperty(None)
    to_settings2 = ObjectProperty(None)
    to_settings3 = ObjectProperty(None)

    drop_area_image = ObjectProperty()
    drop_area_label = ObjectProperty()
    image_src = StringProperty('')

    def __init__(self, **kw):
        super(AvatarGenerateScreen, self).__init__(**kw)
        Window.bind(on_dropfile=self._on_file_drop)

        # Window.bind(on_cursor_enter=self.on_cursor_enter)
        self.image_src = path_root+'/Images/faceset.png'

        self.input_path=""

        self.ids.generate_log.text="ココに画像を入れてください"
        self.event=None


        if not DB.finishGenerate:
            self.event= Clock.schedule_interval(self.log_anime, 1.0)
            self.ids.render_button.background_color=(0.5,0,0,1)
            self.ids.render_button.text="STOP"


    # 画像を読み込む
    def _on_file_drop(self, window, file_path):
        global selected_window
        if DB.selected_window == "avatar_genearate":
            print('dropped anime image')
            # print('dropped image')

            input_path = str(pathlib.Path(str(file_path, 'utf-8').lstrip("b")))
            root, ext = os.path.splitext(input_path)
            self.input_path=input_path
            if ext == '.png' or ext == '.jpg' or ext == '.jpeg':
                print('loading dropped image')
                img = cv2.imread(input_path, cv2.IMREAD_COLOR)

                # external file function
                if anime_face_detect(img):
                    self.drop_area_label.text = ''
                    self.drop_area_image.source = DB.output_path
                    self.drop_area_image.reload()
                    self.ids.render_button.background_color=(0,0.7,0.0,1)
                else:
                    self.drop_area_label.text = '顔が検出されませんでした'
                    self.drop_area_image.source = DB.null_path
                    self.drop_area_image.reload()
                    self.input_path=""
            else:
                self.drop_area_label.text = '画像の読み込みに失敗しました'
                self.input_path=""

                print('->fail')

            return

    def open_folder(self):
        subprocess.Popen(["explorer", r"./"], shell=True)
        return

    def do_prerendering(self):

        if DB.finishGenerate:
            if not self.input_path=="":
                self.ids.render_button.background_color=(0.5,0,0,1)
                self.ids.render_button.text="STOP"
                DB.stopGenerate=False
                DB.finishGenerate=False
                DB.SetSettingImage(self.input_path)
                DB.animator.update_base_image()
                self.event= Clock.schedule_interval(self.log_anime, 1.0)
                # fire_and_forget( self.log_anime)

        elif not DB.finishGenerate:
            self.ids.render_button.background_color=(0.3, 0.3, 0.3, 1.0)
            self.ids.render_button.text="PRERENDERING"
            DB.stopGenerate=True
            DB.finishGenerate=False

        if (DB.finishGenerate and DB.stopGenerate):
            DB.stopGenerate=False
            DB.finishGenerate=True
            self.ids.render_button.background_color=(0.3, 0.3, 0.3, 1.0)
            self.ids.render_button.text="PRERENDERING"
        return


    def log_anime(self,dt):

        if not DB.finishGenerate:
            self.ids.generate_log.text=DB.generate_log
        else:
            DB.finishGenerate=True
            self.ids.render_button.background_color=(0.3, 0.3, 0.3, 1.0)
            self.ids.render_button.text="PRERENDERING"

            if  DB.stopGenerate:
                self.ids.generate_log.text="画像の学習を停止しました"
            else:
                self.ids.generate_log.text="画像の学習に成功しました！"

            DB.stopGenerate=False

            self.event.cancel()
            return False
