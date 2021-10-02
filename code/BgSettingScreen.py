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
Builder.load_file(path_root+'/component/BgSettingsScreen.kv', encoding="utf-8")
#Difine Init Params
INITIAL_WIDTH = DB.INITIAL_WIDTH
INITIAL_HEIGHT = DB.INITIAL_HEIGHT
Window.size = (INITIAL_WIDTH, INITIAL_HEIGHT)





# 背景設定画面
class BgSettingsScreen(Screen):
    popup_close = ObjectProperty(None)
    to_settings0 = ObjectProperty(None)
    to_settings1 = ObjectProperty(None)
    to_settings3 = ObjectProperty(None)

    drop_area_image = ObjectProperty()
    drop_area_label = ObjectProperty()
    image_src = StringProperty('')

    def __init__(self, **kwargs):
        # super().__init__(**kwargs)
        super(BgSettingsScreen, self).__init__(**kwargs)
        Window.bind(on_dropfile=self._on_file_drop)

        # #背景初期
        selectFolder = path_root+"/images/save/bg/"
        folders = [filename for filename in os.listdir(selectFolder) if not filename.startswith('.')]
        folders = natsorted(folders, reverse=True)
        self.image_src = selectFolder + folders[0] + "/bg.png"

        self.image_src = DB.output_bg_path

        #self.drop_area_label.text = 'ファイルをドラッグ＆ドロップ'

        self.create_buttons()
        #self.ids.select_buttons.children[len(folders)-1].text="NOW"
        print("loadedd BG Select Screen")
        self.drop_area_label.text = 'ファイルをドラッグ＆ドロップ'


    # 画像を読み込む
    def _on_file_drop(self, widndow, file_path):
        if DB.selected_window == "bg_settings":
            print('dropped bg image')

            input_bg_path = str(pathlib.Path(str(file_path, 'utf-8').lstrip("b")))
            root, ext = os.path.splitext(input_bg_path)

            if ext == '.png' or ext == '.jpg' or ext == '.jpeg':
                print('loading dropped bg image')

                img = cv2.imread(input_bg_path, cv2.IMREAD_COLOR)
                cv2.imwrite(DB.output_bg_path, img)

                self.drop_area_label.text = ''
                self.drop_area_image.source = DB.output_bg_path
                self.create_save_folder(img)
                # self.reload_images()

            else:
                self.drop_area_label.text = '画像の読み込みに失敗しました'
                print('->fail')

            return

    # セーブデータを選択する
    def select_button(self, num, instance):
        selectFolder = path_root+"/Images/save/bg/"
        folders = [filename for filename in os.listdir(selectFolder) if not filename.startswith('.')]
        folders = natsorted(folders)
        # for i in range(len(folders)):
        #     self.ids.select_buttons.children[i-1].text=""
        # self.ids.select_buttons.children[num-1].text="NOW"

        DB.output_bg_path=selectFolder+folders[num-1]+"/bg.png"
        self.drop_area_image.source = DB.output_bg_path


    #ボタンを配置する
    def create_buttons(self):
        #隠しファイルをスキップしながら格納
        selectFolder = path_root+"/Images/save/bg/"
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
            self.ids.select_buttons.children[-1].color=(1,0,0,1)
            self.ids.select_buttons.children[-1].background_normal= f"{selectFolder}{folders[i]}/bg.png"
            self.ids.select_buttons.children[-1].size_hint_y= 1
            self.ids.select_buttons.children[-1].size_hint_x= None
            self.ids.select_buttons.children[-1].width = Window.size[0] * 0.15
            self.ids.select_buttons.children[-1].id=f"select_btn_{i}"
            self.ids.select_buttons.children[-1].bind(on_release= partial(self.select_button,i+1))

        return

    # セーブデータを追加する
    def create_save_folder(self, img):
        selectFolder = path_root+"/Images/save/bg/"
        folders = [filename for filename in os.listdir(selectFolder) if not filename.startswith('.')]
        folders = natsorted(folders, reverse=True)
        num = int(re.sub("save", "", folders[0])) + 1
        newFolder = selectFolder+"save"+str(num)
        os.mkdir(newFolder)
        cv2.imwrite(newFolder+"/bg.png",img)
        #self.create_button(num)
        self.ids.select_buttons.clear_widgets()
        self.create_buttons()

    # explorerから画像をロードする
    def open_folder(self):
        subprocess.Popen(["explorer", r"./"], shell=True)
        return

    # 選択中のセーブデータを削除する
    def delete_save_folder(self):

        #背景フォルダ削除
        delete_path = DB.output_bg_path.replace('/bg.png', '')
        shutil.rmtree(delete_path)

        #ボタン更新
        self.ids.select_buttons.clear_widgets()
        self.create_buttons()

        #選択している画像を変更
        selectFolder = path_root+"/Images/save/bg/"
        folders = [filename for filename in os.listdir(selectFolder) if not filename.startswith('.')]
        folders = natsorted(folders)
        DB.output_bg_path=selectFolder+folders[len(folders)-1]+"/bg.png"
        self.drop_area_image.source = DB.output_bg_path


