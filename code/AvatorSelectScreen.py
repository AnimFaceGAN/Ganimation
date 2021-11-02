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
Builder.load_file(path_root+'/component/AvatarSelectScreen.kv', encoding="utf-8")
#Difine Init Params
# INITIAL_WIDTH = DB.INITIAL_WIDTH
# INITIAL_HEIGHT = DB.INITIAL_HEIGHT
# Window.size = (INITIAL_WIDTH, INITIAL_HEIGHT)

#TEST REDERING MODE
# DB.renderMode="Low" # "Low"  or  "High"





# アバター選択画面
class AvatarSelectScreen(Screen):
    popup_close = ObjectProperty(None)
    to_settings0 = ObjectProperty(None)
    to_settings2 = ObjectProperty(None)
    to_settings3 = ObjectProperty(None)

    image_src = StringProperty('')
    output_path = ""

    def __init__(self, **kw):
        super(AvatarSelectScreen, self).__init__(**kw)
        # Window.bind(on_dropfile=self._on_file_drop)
        # Window.bind(on_cursor_enter=self.on_cursor_enter)

        #初期選択
        selectFolder = path_root+"/temp/"
        folders = [filename for filename in os.listdir(selectFolder) if not filename.startswith('.')]
        folders = natsorted(folders, reverse=True)
        self.image_src = path_root+"/temp/" + folders[0] + "/thumbnail.png"

        self.create_buttons()
        print("loadedd Select Screen")



    # 画像を読み込む
    def _change_thumnail(self, file_path):
        # print('dropped image')

        input_path = file_path
        root, ext = os.path.splitext(input_path)

        if ext == '.png' or ext == '.jpg' or ext == '.jpeg':
            print('loading dropped image')

            img = cv2.imread(input_path, cv2.IMREAD_COLOR)

            # external file function
            if True:#anime_face_detect(img):
                self.drop_area_image.source = input_path
                self.drop_area_image.reload()
            else:

                print('->fail')
        else:
            print('->fail')

        return

    def select_button(self,num,instance):
        selectFolder = path_root+"/temp/"#"../images/save/face/save"+str(num)+"/"
        #folders=os.listdir(selectFolder)
        folders = [filename for filename in os.listdir(selectFolder) if not filename.startswith('.')]
        folders = natsorted(folders)

        if len(folders)>=num:
            DB.SetBaseImage(folders[num-1])
            DB.animator.change_base_image()
            _path=f"{selectFolder}{folders[num-1]}/thumbnail.png"
            self.output_path = _path
            self._change_thumnail(_path)
        else:
            print("no such data")

    def create_buttons(self):
        selectFolder = path_root+"/temp/"

        #folders=os.listdir(selectFolder)
        #隠しファイルをスキップしながら格納
        folders = [filename for filename in os.listdir(selectFolder) if not filename.startswith('.')]
        folders = natsorted(folders)

        fncs=[]
        for i in range(len(folders)):

            _btn=Button(
                text=f""
                )
            _btn.id=f"select_btn_{i}"

            self.ids.select_buttons.add_widget(_btn,index=i+1)
            # _fnc=lambda x: self.select_button(i+1)
            # fncs.append(_fnc)
            self.ids.select_buttons.children[-1].color=(1,1,1,1)
            self.ids.select_buttons.children[-1].background_normal= f"{selectFolder}{folders[i]}/thumbnail.png"
            self.ids.select_buttons.children[-1].size_hint_y= 1
            self.ids.select_buttons.children[-1].size_hint_x= None
            self.ids.select_buttons.children[-1].width = Window.size[0] * 0.16
            self.ids.select_buttons.children[-1].id=f"select_btn_{i}"
            self.ids.select_buttons.children[-1].bind(on_release= partial(self.select_button,i+1))

        return


    # 選択中のセーブデータを削除する
    def delete_save_folder(self):
        #セーブフォルダ削除
        delete_path = self.output_path.replace("/thumbnail.png", '')
        shutil.rmtree(delete_path)

        #ボタン更新
        self.ids.select_buttons.clear_widgets()
        self.create_buttons()

        #選択している画像を変更
        selectFolder = path_root+"/temp/"
        folders = [filename for filename in os.listdir(selectFolder) if not filename.startswith('.')]
        folders = natsorted(folders)
        self.output_path=selectFolder+folders[len(folders)-1]+"/thumbnail.png"
        self.drop_area_image.source = self.output_path
