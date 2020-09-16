import cv2
import sys
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.widget import Widget
from kivy.graphics.texture import Texture
from kivy.graphics import Rectangle
from kivy.uix.screenmanager import (ScreenManager, Screen, NoTransition, SlideTransition, CardTransition, SwapTransition, FadeTransition, WipeTransition, FallOutTransition, RiseInTransition)
#from kivy.uix.anchorlayout

# 日本語フォント表示対応
from kivy.core.text import LabelBase, DEFAULT_FONT
from kivy.resources import resource_add_path

# Windows用
#resource_add_path('{}\\{}'.format(os.environ['SYSTEMROOT'], 'Fonts'))
#LabelBase.register(DEFAULT_FONT, 'MSGOTHIC.ttc')

resource_add_path('../../Font')
LabelBase.register(DEFAULT_FONT, 'ipaexg.ttf')

Builder.load_file('TutorialScreen.kv')
Builder.load_file('SettingsScreen.kv')
Builder.load_file('VideoScreen.kv')
Builder.load_file('OtherSettingsScreen.kv')

class TutorialScreen(Screen):
    pass

class SettingsScreen(Screen):
    pass

class OtherSettingsScreen(Screen):
    pass

class VideoScreen(Screen):
    pass

# Create the screen manager
#sm = ScreenManager()
sm = ScreenManager(transition = WipeTransition())
sm.add_widget(TutorialScreen(name='tutorial'))
sm.add_widget(SettingsScreen(name='settings'))
sm.add_widget(VideoScreen(name='video'))
sm.add_widget(OtherSettingsScreen(name='other_settings'))

class TestApp(App):
    def build(self):
        return sm

if __name__ == '__main__':
    TestApp().run()
