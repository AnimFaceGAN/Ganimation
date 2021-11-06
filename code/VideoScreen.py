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
from .anime_face_landmark.AnimeFaceDetect import anime_face_detect
from  .Animator.Animator import CreateAnimator
from  database import  *
import cv2
import os.path
import  numpy as np
import time

import numpy as np


#############################################################
#  Kivy関連をimport
#############################################################
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

from kivy.config import Config
Config.set('graphics', 'resizable', False)

##############################################################

#Import Screens
from AvatorGenerateScreen import AvatarGenerateScreen
from AvatorSelectScreen import AvatarSelectScreen
from BgSettingScreen import BgSettingsScreen
from OtherSettingScreen import OtherSettingsScreen
from TurorialScreen import TutorialScreen


import subprocess
from functools import partial

from  copy import deepcopy
from PIL import Image
#for using virtual camera
import pyvirtualcam
import asyncio


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
Builder.load_file(path_root+'/component/VideoScreen.kv', encoding="utf-8")
#Difine Init Params
# INITIAL_WIDTH = DB.INITIAL_WIDTH
# INITIAL_HEIGHT = DB.INITIAL_HEIGHT
# Window.size = (INITIAL_WIDTH, INITIAL_HEIGHT)

def StartUp():
    with open(DB.path_root+"save/setting_data.pickle", mode="rb") as f:
        data = pickle.load(f)

    DB.CAMERA = data["Camera"]
    DB.renderMode = data["RenderingMode"]



# ビデオ画面
class VideoScreen(Screen):

    bg = ObjectProperty()
    anime = ObjectProperty()
    bg_src = StringProperty('')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        StartUp()

        #背景初期
        selectFolder = path_root+"/Images/save/bg/"
        folders = [filename for filename in os.listdir(selectFolder) if not filename.startswith('.')]
        folders = natsorted(folders, reverse=True)
        self.bg_src = selectFolder + folders[0] + "/bg.png"
        _output_bg_path = self.bg_src

        self.capture = cv2.VideoCapture(DB.CAMERA)


        Clock.schedule_interval(self.update, 0.05)

        W,H=800,800#pyautogui.size()
        self.cam=pyvirtualcam.Camera(width=W, height=H, fps=10)

        #Init Video Back Ground Images
        self.video_bg_path=_output_bg_path

        self.video_bg=cv2.imread(self.video_bg_path)
        self.video_bg=cv2.resize(self.video_bg,(W,H))
        self.video_bg=cv2.cvtColor(self.video_bg, cv2.COLOR_BGRA2RGBA)

        #Init Screen Settings
        # self._init_screen()

        # Clock.schedule_interval(self.update, 0.01)
        print('init video')
        DB.playing_video = True


    def stop_video(self):
        DB.playing_video = False
        # self.capture.release()
        print('stop video')

    def start_video(self):
        DB.playing_video = True
        # self.capture.release()
        print('start video')

    def update(self, dt):


        start=time.time()
        if not DB.playing_video:
            return
        # print("video now")
        ret, self.frame = self.capture.read()

        ###############################################
        # 制限機能の追加
        ###############################################
        if DB.limitApp.check():
            self.ids.anime.source = path_root+"/Images/null.png"
            self.bg_src = path_root+"/Images/thankyou.png"
            return

        if ret == False:
            print("No Signal")
            self.ids.message.text = "No Signal"
            self.ids.anime.source = path_root+"/Images/null.png"
            return
        self.ids.message.text = ""



        # リアル顔画像をデータベースにセット
        DB.SetRealFaces(self.frame)
        result = DB.animator.update_image()

        if result == None:
            return
        _, flg = result
        if not flg:
            return

        #アニメ顔画像のデータベースから取得
        self.animeface = DB.GetAnimeFaces()
        self.animeface =cv2.resize(self.animeface, (500, 500))


        # Kivy Textureに変換

        buf = cv2.flip(self.animeface, -1).tobytes()
        texture = Texture.create(size=(self.animeface.shape[1], self.animeface.shape[0]), colorfmt='rgba')
        texture.blit_buffer(buf, colorfmt='rgba', bufferfmt='ubyte')
        # インスタンスのtextureを変更
        self.anime.texture = texture

        ##############################################################################
        #----  Virtual Camera  ----
        ##############################################################################
        if DB.virtual_camera:
            W,H=800,800#pyautogui.size()

            #Change bg if it chenged
            if self.video_bg_path!=DB.output_bg_path:
                print("changed")
                self.video_bg_path= DB.output_bg_path
                self.video_bg=cv2.imread(self.video_bg_path)
                self.video_bg=cv2.resize(self.video_bg,(W,H))
                self.video_bg=cv2.cvtColor(self.video_bg, cv2.COLOR_BGRA2RGBA)

            # bg=cv2.resize(self.video_bg,(W,H))
            bg=self.video_bg
            # bg=cv2.cvtColor(bg, cv2.COLOR_BGRA2RGBA)
            bg=Image.fromarray(bg).convert('RGBA')
            anime=Image.fromarray(self.animeface)
            img_clear = Image.new("RGBA", bg.size, (255, 255, 255, 0))

            anime_h, anime_w = anime.size[:2]
            bg_h, bg_w = bg.size[:2]

            img_clear.paste(anime, (int((bg_h-anime_h)/2), int((bg_w-anime_w)/2)))
            bg = Image.alpha_composite(bg, img_clear)
            _frame=np.array(bg)[:,:,:3]#cv2.cvtColor(np.array(bg), cv2.COLOR_RGBA2RGB)

            self.cam.send(_frame)
            elapsed_time = time.time() - start
            # print(f"\r FPS : {round(1/elapsed_time,2)} [frame/sec]" ,end="")
        ##############################################################################

    # チュートリアルポップアップ表示
    def tutorial_popup_open(self):
        self.stop_video()

        content = TutorialScreen(popup_close=self.popup_close)
        self.popup = Popup(title='',separator_height=0, content=content, size_hint=(0.9, 0.9), auto_dismiss=False)
        self.popup.open()

    # 設定ポップアップ表示
    def settings_popup_open(self):
        DB.selected_window = "avatar_genearate"
        self.stop_video()

        content = AvatarGenerateScreen(popup_close=self.popup_close, to_settings1=self.to_settings1,  to_settings2=self.to_settings2, to_settings3=self.to_settings3)
        self.popup = Popup(title='',separator_height=0, content=content, size_hint=(0.9, 0.9), auto_dismiss=False)

        self.popup.open()


    # アバター生成へ画面遷移
    def to_settings0(self):
        DB.selected_window = "avatar_genearate"
        self.stop_video()

        self.popup.dismiss()
        content = AvatarGenerateScreen(popup_close=self.popup_close, to_settings1=self.to_settings1, to_settings2=self.to_settings2, to_settings3=self.to_settings3)
        self.popup = Popup(title='',separator_height=0, content=content, size_hint=(0.9, 0.9), auto_dismiss=False)
        self.popup.open()

    # アバター選択へ画面遷移
    def to_settings1(self):
        DB.selected_window = "avatar_select"
        self.stop_video()

        self.popup.dismiss()
        content = AvatarSelectScreen(popup_close=self.popup_close, to_settings0=self.to_settings0, to_settings2=self.to_settings2, to_settings3=self.to_settings3)
        self.popup = Popup(title='',separator_height=0, content=content, size_hint=(0.9, 0.9), auto_dismiss=False)
        self.popup.open()

    # 背景設定へ画面遷移
    def to_settings2(self):
        DB.selected_window = "bg_settings"

        self.popup.dismiss()
        content = BgSettingsScreen(popup_close=self.popup_close, to_settings0=self.to_settings0, to_settings1=self.to_settings1, to_settings3=self.to_settings3)
        self.popup = Popup(title='',separator_height=0, content=content, size_hint=(0.9, 0.9), auto_dismiss=False)
        self.popup.open()

    # その他設定へ画面遷移
    def to_settings3(self):
        DB.selected_window = "other_settings"
        self.popup.dismiss()
        content = OtherSettingsScreen(popup_close=self.popup_close, to_settings0=self.to_settings0, to_settings1=self.to_settings1, to_settings2=self.to_settings2)
        self.popup = Popup(title='',separator_height=0, content=content, size_hint=(0.9, 0.9), auto_dismiss=False)
        self.popup.open()

    # ポップアップを閉じる
    def popup_close(self):
        DB.selected_window = "video"
        self.bg_src = DB.output_bg_path #背景のパスを指定
        self.start_video()
        self.popup.dismiss()
        self.bg.reload() #背景更新

        self.capture = cv2.VideoCapture(DB.CAMERA)
        self.popup=None
        # self.capture = cv2.VideoCapture(CAMERA)
