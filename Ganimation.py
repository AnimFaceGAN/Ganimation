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
