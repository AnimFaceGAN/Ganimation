#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
import os

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

from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.uix.screenmanager import (ScreenManager, Screen, NoTransition, SlideTransition, CardTransition,
                                    SwapTransition, FadeTransition, WipeTransition, FallOutTransition, RiseInTransition)

#for pupup
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout

# 日本語フォント表示対応
from kivy.core.text import LabelBase, DEFAULT_FONT
from kivy.resources import resource_add_path
from  Animator.Animator import CreateAnimator

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
import pyautogui
def obs_camera(anime,bg,cam):
    # s=time.time()
    W,H=800,800#pyautogui.size()
    bg=cv2.resize(bg,(W,H))
    bg=cv2.cvtColor(bg, cv2.COLOR_BGRA2RGBA)
    anime=cv2.resize(anime,dsize=None,fx=1, fy=1)
    # anime=cv2.cvtColor(anime, cv2.COLOR_BGRA2RGBA)


    bg=Image.fromarray(bg).convert('RGBA')
    anime=Image.fromarray(anime)
    img_clear = Image.new("RGBA", bg.size, (255, 255, 255, 0))

    anime_h, anime_w = anime.size[:2]
    bg_h, bg_w = bg.size[:2]

    img_clear.paste(anime, (int((bg_h-anime_h)/2), int((bg_w-anime_w)/2)))
    bg = Image.alpha_composite(bg, img_clear)

    # x1, y1, x2, y2 = 0, 0, anime.shape[1], anime.shape[0]

    # bg[y1:y2, x1:x2] = bg[y1:y2, x1:x2] * (1 - anime[:, :, 3:] / 255) + \
    #                   anime[:, :, :3] * (anime[:, :, 3:] / 255)

    _frame=cv2.cvtColor(np.array(bg), cv2.COLOR_RGBA2RGB)
    # _frame=cv2.resize(_frame,(W,H))
    # cv2.imwrite("./obs_frame.png",np.array(bg))
    cam.send(_frame)
    # print(time.time()-s)
    # cam.sleep_until_next_frame()
    return

animator=CreateAnimator()
#animator.update_image().show()

# フォント読み込み Windows用
resource_add_path('{}\\{}'.format(os.environ['SYSTEMROOT'], 'Fonts'))
LabelBase.register(DEFAULT_FONT, 'MSGOTHIC.ttc')

# フォント読み込み Mac用
#resource_add_path('./Font')
#LabelBase.register(DEFAULT_FONT, 'ipaexg.ttf')
# フォント読み込み
resource_add_path('./Font')
LabelBase.register(DEFAULT_FONT, 'ipaexg.ttf')

# フォント読み込み Windows用
#resource_add_path('{}\\{}'.format(os.environ['SYSTEMROOT'], 'Fonts'))
#LabelBase.register(DEFAULT_FONT, 'MSGOTHIC.ttc')

# Kivyファイルの読み込み
Builder.load_file('TutorialScreen.kv', encoding="utf-8")
Builder.load_file('SettingsScreen.kv', encoding="utf-8")
Builder.load_file('VideoScreen.kv', encoding="utf-8")
Builder.load_file('OtherSettingsScreen.kv', encoding="utf-8")

# アニメ顔画像のパス
output_path = '../images/output.png'
null_path = '../images/faceset.png'  # 画像未入力時に表示する

# DataBaseのインスタンス化
#DataLoarderクラスのインスタンス化
loarder=DataLoarder()
#それぞれのインスタンスを読み込む
f2b=loarder.create_foward2back()
b2f=loarder.create_back2foward()
DB=loarder.create_database()

# 背景画像のパス
output_bg_path = "../images/save/bg/save1/bg.png"
selected_bg = 1

#ビデオ画面のupdateオンオフ
playing_video = False
selected_window = "video"

# チュートリアル画面
class TutorialScreen(Screen):
    popup_close = ObjectProperty(None)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def start_video(self):
        # global playing_video
        # playing_video = True
        print('start')


# 画像設定画面
class SettingsScreen(Screen):
    popup_close = ObjectProperty(None)
    to_settings2 = ObjectProperty(None)

    drop_area_image = ObjectProperty()
    drop_area_label = ObjectProperty()
    image_src = StringProperty('')

    def __init__(self, **kw):
        super(SettingsScreen, self).__init__(**kw)
        Window.bind(on_dropfile=self._on_file_drop)
        # Window.bind(on_cursor_enter=self.on_cursor_enter)
        self.image_src = '../images/faceset.png'

    # 画像を読み込む
    def _on_file_drop(self, window, file_path):
        # print('dropped image')

        input_path = str(pathlib.Path(str(file_path, 'utf-8').lstrip("b")))
        root, ext = os.path.splitext(input_path)

        if ext == '.png' or ext == '.jpg' or ext == '.jpeg':
            print('loading dropped image')

            img = cv2.imread(input_path, cv2.IMREAD_COLOR)

            # external file function
            if anime_face_detect(img):
                self.drop_area_label.text = ''
                self.drop_area_image.source = output_path
                self.drop_area_image.reload()
                DB.SetSettingImage(input_path)
                animator.update_base_image()
            else:
                self.drop_area_label.text = '顔が検出されませんでした'
                self.drop_area_image.source = null_path
                self.drop_area_image.reload()
        else:
            self.drop_area_label.text = '画像の読み込みに失敗しました'
            print('->fail')

        return


    def start_video(self):
        global playing_video
        playing_video = True
        print('start')
    
    def select_button(self, num):
        global output_bg_path
        print(num)
        selectFolder = "../temp/"#"../images/save/face/save"+str(num)+"/"
        folders=os.listdir(selectFolder)
        # if not os.listdir(folders):
        #     print("empty")
        # else:
        #     print("not empty")
        print(folders)
        if len(folders)>=num:
            DB.SetBaseImage(folders[num-1])
            animator.change_base_image()
        else:
            print("no such data")



# 詳細設定画面
class OtherSettingsScreen(Screen):
    popup_close = ObjectProperty(None)
    to_settings1 = ObjectProperty(None)

    drop_area_image = ObjectProperty()
    drop_area_label = ObjectProperty()
    save1 = ObjectProperty()
    save2 = ObjectProperty()
    save3 = ObjectProperty()
    save4 = ObjectProperty()
    save5 = ObjectProperty()
    save6 = ObjectProperty()

    save1_src = StringProperty('')
    save2_src = StringProperty('')
    save3_src = StringProperty('')
    save4_src = StringProperty('')
    save5_src = StringProperty('')
    save6_src = StringProperty('')
    image_src = StringProperty('')

    def __init__(self, **kwargs):
        # super().__init__(**kwargs)
        super(OtherSettingsScreen, self).__init__(**kwargs)
        Window.bind(on_dropfile=self._on_file_drop)
        self.image_src = output_bg_path
        self.drop_area_label.text = 'ファイルをドラッグ＆ドロップ'
        self.save1_src = "../images/save/bg/save1/bg.png"
        self.save2_src = "../images/save/bg/save2/bg.png"
        self.save3_src = "../images/save/bg/save3/bg.png"
        self.save4_src = "../images/save/bg/save4/bg.png"
        self.save5_src = "../images/save/bg/save5/bg.png"
        self.save6_src = "../images/save/bg/save6/bg.png"

    def start_video(self):
        global playing_video
        playing_video = True
        print('start')

    # 画像を読み込む
    def _on_file_drop(self, widndow, file_path):
        global selected_window
        if selected_window == "other_settings":
            print('dropped bg image')

            input_bg_path = str(pathlib.Path(str(file_path, 'utf-8').lstrip("b")))
            root, ext = os.path.splitext(input_bg_path)

            if ext == '.png' or ext == '.jpg' or ext == '.jpeg':
                print('loading dropped bg image')

                img = cv2.imread(input_bg_path, cv2.IMREAD_COLOR)
                cv2.imwrite(output_bg_path, img)

                self.drop_area_label.text = ''
                self.drop_area_image.source = output_bg_path
                # self.reload_images()

            else:
                self.drop_area_label.text = '画像の読み込みに失敗しました'
                print('->fail')

            return

    def select_button(self, num):
        global output_bg_path
        _ori_path="../Images/save/bg/"
        _folders=os.listdir(_ori_path)
        if len(_folders)>=num:
            output_bg_path=_ori_path+_folders[num-1]+"/bg.png"

CAMERA=1
# ビデオ画面
class VideoScreen(Screen):

    bg = ObjectProperty()
    anime = ObjectProperty()
    bg_src = StringProperty('')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bg_src = "../images/save/bg/save1/bg.png"
        self.capture = cv2.VideoCapture(CAMERA)
        self.animator = animator

        Clock.schedule_interval(self.update, 0.05)

        W,H=800,800#pyautogui.size()
        self.cam=pyvirtualcam.Camera(width=W, height=H, fps=10) 

        self.video_bg_path=output_bg_path
        self.video_bg=cv2.imread(self.video_bg_path)

        # Clock.schedule_interval(self.update, 0.01)
        print('init video')
        global playing_video
        playing_video = True

    def stop_video(self):
        global playing_video
        playing_video = False
        self.capture.release()
        print('stop video')

    def start_video(self):
        global playing_video
        playing_video = True
        self.capture.release()
        print('start video')

    def update(self, dt):
        start=time.time()

        if not playing_video:
            return
        # print("video now")
        ret, self.frame = self.capture.read()
        # print(ret)

        # リアル顔画像をデータベースにセット
        DB.SetRealFaces(self.frame)

        result = self.animator.update_image()

        if result == None:
            return
        _, flg = result
        if not flg:
            return
        # print("animation")
        

        #アニメ顔画像のデータベースから取得
        self.animeface = DB.GetAnimeFaces()
        self.animeface =cv2.resize(self.animeface, (500, 500))

        # cv2.imwrite("./screen_cap.png",self.animeface)

        # cv2.imshow("",self.animeface)
        # cv2.waitKey(4)

        # デバッグ用
        # self.animeface = self.frame
        # self.animeface = cv2.resize(self.frame, (280, 280))

        # Kivy Textureに変換
        # buf = cv2.flip(self.animeface, -1).tobytes()
        # texture = Texture.create(size=(self.animeface.shape[1], self.animeface.shape[0]), colorfmt='rgba')
        # texture.blit_buffer(buf, colorfmt='rgba', bufferfmt='ubyte')
        # # インスタンスのtextureを変更
        # self.texture = texture
        
        # Kivy Textureに変換
        buf = cv2.flip(self.animeface, -1).tobytes()
        texture = Texture.create(size=(self.animeface.shape[1], self.animeface.shape[0]), colorfmt='rgba')
        texture.blit_buffer(buf, colorfmt='rgba', bufferfmt='ubyte')
        # インスタンスのtextureを変更
        self.anime.texture = texture
        
        ##############################################################################
        #----  Virtual Camera  ----
        ##############################################################################
        #Change bg if it chenged 
        if self.video_bg_path!=output_bg_path:
            print("changed")
            self.video_bg_path= output_bg_path
            self.video_bg=cv2.imread(self.video_bg_path)
        start=time.time()
        # _bg=cv2.imread(output_bg_path)
        # obs_camera(self.animeface,_bg,self.cam)

        W,H=800,800#pyautogui.size()
        bg=cv2.resize(self.video_bg,(W,H))
        bg=cv2.cvtColor(bg, cv2.COLOR_BGRA2RGBA)
        # anime=cv2.resize(self.animeface,dsize=None,fx=1, fy=1)
        # anime=cv2.cvtColor(anime, cv2.COLOR_BGRA2RGBA)



        bg=Image.fromarray(bg).convert('RGBA')
        anime=Image.fromarray(self.animeface)
        img_clear = Image.new("RGBA", bg.size, (255, 255, 255, 0))


        anime_h, anime_w = anime.size[:2]
        bg_h, bg_w = bg.size[:2]

        img_clear.paste(anime, (int((bg_h-anime_h)/2), int((bg_w-anime_w)/2)))
        bg = Image.alpha_composite(bg, img_clear)



        # x1, y1, x2, y2 = 0, 0, anime.shape[1], anime.shape[0]

        # bg[y1:y2, x1:x2] = bg[y1:y2, x1:x2] * (1 - anime[:, :, 3:] / 255) + \
        #                   anime[:, :, :3] * (anime[:, :, 3:] / 255)

        _frame=cv2.cvtColor(np.array(bg), cv2.COLOR_RGBA2RGB)

        # _frame=cv2.resize(_frame,(W,H))
        # cv2.imwrite("./obs_frame.png",np.array(bg))
        self.cam.send(_frame)
        elapsed_time = time.time() - start
        print(f"\r FPS : {round(1/elapsed_time,2)} [frame/sec]" ,end="")


        ##############################################################################

        

    # チュートリアルポップアップ表示
    def tutorial_popup_open(self):
        self.stop_video()

        content = TutorialScreen(popup_close=self.popup_close)
        self.popup = Popup(title='',separator_height=0, content=content, size_hint=(0.7, 0.7), auto_dismiss=False)
        self.popup.open()

    # 設定ポップアップ表示
    def settings_popup_open(self):
        global selected_window
        selected_window = "settings"
        self.stop_video()

        content = SettingsScreen(popup_close=self.popup_close, to_settings2=self.to_settings2)
        self.popup = Popup(title='',separator_height=0, content=content, size_hint=(0.7, 0.7), auto_dismiss=False)
        self.popup.open()

    # 顔イラスト設定へ画面遷移
    def to_settings1(self):
        global selected_window
        selected_window = "settings"

        self.popup.dismiss()
        content = SettingsScreen(popup_close=self.popup_close, to_settings2=self.to_settings2)
        self.popup = Popup(title='',separator_height=0, content=content, size_hint=(0.7, 0.7), auto_dismiss=False)
        self.popup.open()

    # 背景設定へ画面遷移
    def to_settings2(self):
        global selected_window
        selected_window = "other_settings"

        self.popup.dismiss()
        content = OtherSettingsScreen(popup_close=self.popup_close, to_settings1=self.to_settings1)
        self.popup = Popup(title='',separator_height=0, content=content, size_hint=(0.7, 0.7), auto_dismiss=False)
        self.popup.open()

    def popup_close(self):
        global selected_window
        selected_window = "video"
        self.bg_src = output_bg_path #背景のパスを指定
        self.start_video()
        self.popup.dismiss()
        self.bg.reload() #背景更新
        self.capture = cv2.VideoCapture(CAMERA)



# class VideoManager(Image):

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.is_animation = False
#         self.capture = cv2.VideoCapture(1)
#         self.animator = animator

#         Clock.schedule_interval(self.update, 0.07)

#     def start_animation(self):
#         self.is_animation = True
#         self.capture = cv2.VideoCapture(1)

#     def stop_animation(self):
#         self.is_animating = False
#         self.cap.release()

#     def update(self, dt):
#         if not playing_video:
#             return

#         ret, self.frame = self.capture.read()

#         # リアル顔画像をデータベースにセット
#         DB.SetRealFaces(self.frame)

#         # アニメ顔画像のデータベースから取得
#         # self.animeface = DB.GetAnimeFaces()

#         # ビデオ表示テスト
#         # アニメ顔画像のデータベースから取得
#         # self.animeface = DB.GetAnimeFaces()
#         result = self.animator.update_image()
#         if result == None:
#             return
#         _, flg = result
#         if not flg:
#             return
#         animeface=DB.GetAnimeFaces()
#         self.animeface =cv2.resize(animeface, (500, 500))

#         #self.animeface = self.frame

#         # Kivy Textureに変換
#         buf = cv2.flip(self.animeface, -1).tobytes()
#         texture = Texture.create(size=(self.animeface.shape[1], self.animeface.shape[0]), colorfmt='rgba')
#         texture.blit_buffer(buf, colorfmt='rgba', bufferfmt='ubyte')
#         # インスタンスのtextureを変更
#         self.texture = texture

# def show_img(img, title=""):
#     cv2.imshow(title, img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()



sm = ScreenManager(transition=WipeTransition())
# sm.add_widget(TutorialScreen(name='tutorial'))
# sm.add_widget(SettingsScreen(name='settings'))
sm.add_widget(VideoScreen(name='video'))
# sm.add_widget(OtherSettingsScreen(name='other_settings'))

class GanimationApp(App):
    def build(self):
        return sm


if __name__ == '__main__':
    GanimationApp().run()
