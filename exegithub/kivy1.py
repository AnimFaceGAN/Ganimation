from kivy.app import App
from kivy.properties import ObjectProperty, StringProperty
from kivy.uix.widget import Widget
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
class DeepPlayer(BoxLayout):
    fn_video = StringProperty()
    def __init__(self):
        super().__init__()
    # ファイル選択
    def dismiss_popup(self):
        self._popup.dismiss()
    def load(self, filename):
        self.fn_video = filename[0]
        self.dismiss_popup()
    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(.8, .8))
        self._popup.open()
class DeepPlayerApp(App):
    def __init__(self):
        super().__init__()
        self.title = "Deep Learning Video Player"
    def build(self):
        return DeepPlayer()
if __name__ == '__main__':
    DeepPlayerApp().run()

