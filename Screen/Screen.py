import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import pathlib
from anime_face_landmark.AnimeFaceDetect import anime_face_detect
from  database import  *
import cv2
import os.path


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

# 日本語フォント表示対応
from kivy.core.text import LabelBase, DEFAULT_FONT
from kivy.resources import resource_add_path
#from  Animator.Animator import CreateAnimator

#animator=CreateAnimator()
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
loarder=GetDataLoarder()
#それぞれのインスタンスを読み込む
f2b=loarder.create_foward2back()
b2f=loarder.create_back2foward()
DB=loarder.create_database()


# チュートリアル画面
class TutorialScreen(Screen):
    pass


# 画像設定画面
class SettingsScreen(Screen):
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
                DB.SetSettingImage(output_path)
            else:
                self.drop_area_label.text = '顔が検出されませんでした'
                self.drop_area_image.source = null_path
                self.drop_area_image.reload()
        else:
            self.drop_area_label.text = '画像の読み込みに失敗しました'
            print('->fail')

        return

    # ビデオ画面へ移ると同時にカメラ起動
    def to_video(self):
        #VideoManager.start_animation()
        pass


# 詳細設定画面
class OtherSettingsScreen(Screen):
    pass

# ビデオ画面
class VideoScreen(Screen):
    pass

class VideoManager(Image):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_animation = False
        self.capture = cv2.VideoCapture(1)
        Clock.schedule_interval(self.update, 0.01)

    def start_animation(self):
        self.is_animation = True
        self.capture = cv2.VideoCapture(1)

    def off_animation(self):
        self.is_animating = False
        self.cap.release()

    def update(self, dt):
        #if self.is_animation == True: #上手くいかないので一旦コメントアウト
            ret, self.frame = self.capture.read()

            # リアル顔画像をデータベースにセット
            DB.SetRealFaces(self.frame)

            #アニメ顔画像のデータベースから取得
            #self.animeface = DB.GetAnimeFaces()
            self.animeface = self.frame #ビデオ表示テスト

            # Kivy Textureに変換
            buf = cv2.flip(self.animeface, -1).tostring()
            texture = Texture.create(size=(self.animeface.shape[1], self.animeface.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # インスタンスのtextureを変更
            self.texture = texture


    '''
    image_texture = ObjectProperty(None)
    is_animating = False
    capture = None
    frame = None
    texture = None
    count = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Clock.schedule_interval(self.update, 0.01)
        Clock.schedule_interval(self.update_anime, 0.01)
        Clock.schedule_interval(self.update_display, 0.01)

    # アニメーションを開始する
    @classmethod
    def on_animation(cls):
        cls.capture = cv2.VideoCapture(1)
        cls.is_animating = True

    # アニメーションを終了する
    @classmethod
    def off_animation(cls):
        cls.cap.release()
        cls.is_animating = False

    @classmethod
    def update(cls, dt):
        if cls.is_animating:
            ret, cls.frame = cls.capture.read()

            # リアル顔画像をデータベースにセット
            DB.SetRealFaces(cls.frame)

    @classmethod
    def update_anime(cls, dt):
        # フレームを読み込み
        if cls.is_animating:
            # Kivy Textureに変換
            buf = cv2.flip(cls.frame, 0).tostring()
            texture = Texture.create(size=(cls.frame.shape[1], cls.frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # インスタンスのtextureを変更
            cls.texture = texture
            cls.update_display(texture)

    def update_display(self, texture):
        self.image_texture.texture = texture
    '''


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
