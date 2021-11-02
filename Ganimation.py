from kivy import Config
Config.set('graphics', 'width', '1200')#1200
Config.set('graphics', 'height', '800')#800
Config.set('graphics', 'minimum_width', '700')
Config.set('graphics', 'minimum_height', '400')
Config.set('input', 'mouse', 'mouse,disable_multitouch')
Config.set('kivy','window_icon', "./resource/Images/icon/image.ico")

from kivy.app import App
from kivy.uix.screenmanager import ScreenManager , WipeTransition

from code.VideoScreen import VideoScreen


sm = ScreenManager(transition=WipeTransition())
sm.add_widget(VideoScreen(name='video'))

class GanimationApp(App):
    def build(self):
        return sm


if __name__ == '__main__':
    GanimationApp().run()
