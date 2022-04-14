# from Code.face_mesh_mediapipe import MediaPipe_Method
# from GCM.Geometric_computation import Geometric_Computation
from kivymd.app import MDApp
from kivymd.uix.screen import Screen
from kivy.uix.screenmanager import ScreenManager
from kivy.core.window import Window
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.gridlayout import MDGridLayout
from kivymd.uix.button import MDRaisedButton, MDRectangleFlatButton
from kivymd.uix.label import MDLabel
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.uix.widget import Widget
import cv2


Window.size = (90*6, 160*6)    # remove this for deployment


class MainScreen(Screen):
    def load_frame(self):
        self.capture = cv2.VideoCapture(1)
        Clock.schedule_interval(self.load_video, 1.0/30.0)

    def load_video(self, *args):
        # read frame from opencv
        ret, frame = self.capture.read()
        frame = frame[120:120 + 250, 200:200 + 250, :]
        self.frame = frame

        # Flip horizontal and convert image to texture
        buffer = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.ids.web_feed.texture = texture


class OptionsScreen(Screen):
    pass


class MyApp(MDApp):

    def build(self):
        self.theme_cls.primary_palette = 'Green'
        self.theme_cls.primary_hue = '400'
        self.web_feed = Image()
        return


if __name__ == '__main__':
    MyApp().run()