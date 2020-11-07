import pathlib
from anime_face_landmark.AnimeFaceDetect import anime_face_detect
import cv2
import os.path
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.widget import Widget
from kivy.graphics.texture import Texture
from kivy.graphics import Rectangle
from kivy.properties import StringProperty, ObjectProperty
from kivy.clock import Clock
from kivy.config import Config
from kivy.core.window import Window
from kivy.uix.screenmanager import (ScreenManager, Screen, NoTransition, SlideTransition, CardTransition, SwapTransition, FadeTransition, WipeTransition, FallOutTransition, RiseInTransition)
# 日本語フォント表示対応
from kivy.core.text import LabelBase, DEFAULT_FONT
from kivy.resources import resource_add_path
from  Animator import CreateAnimator

#animator=CreateAnimator()
#animator.update_image().show()

# フォント読み込み Windows用
resource_add_path('{}\\{}'.format(os.environ['SYSTEMROOT'], 'Fonts'))
LabelBase.register(DEFAULT_FONT, 'MSGOTHIC.ttc')

# フォント読み込み Mac用
#resource_add_path('./Font')
#LabelBase.register(DEFAULT_FONT, 'ipaexg.ttf')

# Kivyファイルの読み込み
Builder.load_file('TutorialScreen.kv')
Builder.load_file('SettingsScreen.kv')
Builder.load_file('VideoScreen.kv')
Builder.load_file('OtherSettingsScreen.kv')

# アニメ顔画像のパス
output_path = '../images/output.png'
null_path = '../images/null.png'

class TutorialScreen(Screen):
    pass


class SettingsScreen(Screen):
    drop_area_image = ObjectProperty()
    drop_area_label = ObjectProperty()
    image_src = StringProperty('')

    def __init__(self, **kw):
        super(SettingsScreen, self).__init__(**kw)
        Window.bind(on_dropfile=self._on_file_drop)
        #Window.bind(on_cursor_enter=self.on_cursor_enter)
        self.image_src = '../images/null.png'


    # load image function
    def _on_file_drop(self, window, file_path):
        print('dropped image')

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
            else:
                self.drop_area_label.text = '顔が検出されませんでした'
                self.drop_area_image.source = null_path
                self.drop_area_image.reload()
        else:
            self.drop_area_label.text = '画像の読み込みに失敗しました'
            print('->fail')

        return



class OtherSettingsScreen(Screen):
    pass


class VideoScreen(Screen):
    pass

sm = ScreenManager(transition=WipeTransition())
sm.add_widget(TutorialScreen(name='tutorial'))
sm.add_widget(SettingsScreen(name='settings'))
sm.add_widget(VideoScreen(name='video'))
sm.add_widget(OtherSettingsScreen(name='other_settings'))

class TestApp(App):
    def build(self):
        return sm

if __name__ == '__main__':
    TestApp().run()
