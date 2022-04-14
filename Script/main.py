from mediapipe.framework.formats import location_data_pb2
from Code.face_mesh_mediapipe import MediaPipe_Method
from GCM.geometric_computation import Geometric_Computation
from kivymd.app import MDApp
from kivy.lang import Builder
from kivymd.uix.screen import Screen
from kivymd.uix.tab import MDTabsBase, MDTabs
from kivy.uix.screenmanager import ScreenManager
from kivy.core.window import Window
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.floatlayout import MDFloatLayout
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivymd.uix.behaviors import FakeRectangularElevationBehavior
from kivy.clock import Clock
from kivy.resources import resource_add_path
import cv2
import os
import math
import mediapipe as mp

resource_add_path('.')

mp_face_detection = mp.solutions.face_detection

mp_drawing = mp.solutions.drawing_utils

Window.size = (90 * 6, 160 * 6)  # remove this for deployment


class MainScreen(Screen):
    pass


class UserProfile(Screen):
    pass


class NavBar(FakeRectangularElevationBehavior, MDFloatLayout):
    pass


class ExamPage(Screen, MDTabsBase, MDFloatLayout):
    pass


class Tab(MDTabsBase):
    pass


class DataWindow(Screen):
    pass


class MPFaceMesh(Screen, Image, MDBoxLayout):

    def __init__(self, **kw):
        super().__init__(**kw)
        self.frame = None
        self.MP = None

        self.capture = None
        self.image1 = None
        self.image2 = None
        self.image3 = None
        self.image4 = None
        self.image5 = None
        self.image6 = None

        self.result1 = None
        self.result2 = None
        self.result3 = None
        self.result4 = None
        self.result5 = None
        self.result6 = None

        self.mp_result1 = None
        self.mp_result2 = None
        self.mp_result3 = None
        self.mp_result4 = None
        self.mp_result5 = None
        self.mp_result6 = None

        self.refs = [127, 356]
        self.num_of_pics = 0

        self.results = None

    def load_frame(self, id, root):
        self.root = root
        self.capture = cv2.VideoCapture(0)
        self.clock = Clock.schedule_interval(self.load_video, 1.0 / 30.0)

    def detect_face(self):
        with mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5) as face_detection:
            while self.capture.isOpened():
                success, image = self.capture.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_detection.process(image)

                if results.detections:
                    return results.detections[0]
                else:
                    return None
                # # Flip the image horizontally for a selfie-view display.
                # cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
                # if cv2.waitKey(5) & 0xFF == 27:
                #     break
        # self.capture.release()

    def _normalized_to_pixel_coordinates(self, normalized_x, normalized_y, image_width, image_height):
        """Converts normalized value pair to pixel coordinates."""

        # Checks if the float value is between 0 and 1.
        def is_valid_normalized_value(value: float) -> bool:
            return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                              math.isclose(1, value))

        if not (is_valid_normalized_value(normalized_x) and
                is_valid_normalized_value(normalized_y)):
            # TODO: Draw coordinates even if it's outside of the image bounds.
            return None
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return x_px, y_px

    def get_face_box(self, image, detection):
        if not detection.location_data:
            return
        if image.shape[2] != 3:
            raise ValueError('Input image must contain three channel rgb data.')
        image_rows, image_cols, _ = image.shape

        location = detection.location_data
        if location.format != location_data_pb2.LocationData.RELATIVE_BOUNDING_BOX:
            raise ValueError(
                'LocationData must be relative for this drawing funtion to work.')

        # Draws bounding box if exists.
        if not location.HasField('relative_bounding_box'):
            return
        relative_bounding_box = location.relative_bounding_box
        rect_start_point = self._normalized_to_pixel_coordinates(
            relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
            image_rows)
        rect_end_point = self._normalized_to_pixel_coordinates(
            relative_bounding_box.xmin + relative_bounding_box.width,
            relative_bounding_box.ymin + relative_bounding_box.height, image_cols,
            image_rows)
        return rect_start_point, rect_end_point

    def load_video(self, *args):
        # read frame from opencv
        if self.capture is None:
            self.clock.cancel()
            self.capture.release()

        box = self.detect_face();
        ret, frame = self.capture.read()
        if box is not None:
            start, end = self.get_face_box(frame, detection=box)
            if start is not None and end is not None:
                sx, sy = start
                ex, ey = end
                self.frame = frame[sy:ey, sx: ex]

                # Flip horizontal and convert image to texture
                buffer = cv2.flip(self.frame, 0).tobytes()
                texture = Texture.create(size=(self.frame.shape[1], self.frame.shape[0]), colorfmt='bgr')
                texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
                self.root.ids['web_feed'].texture = texture
        else:
            print('face out of frame')

    def take_picture(self, index):
        # ensuring directory for later deletion
        directory = r'../temp_images'
        os.chdir(directory)
        # keep track of number of pics taken
        self.num_of_pics += 1
        # creating new image source
        name = "temp_image" + str(self.num_of_pics) + ".jpg"

        # getting index
        id_name = 'image' + str(index + 1)

        # updating frame
        frame = self.frame

        if index == 0:
            self.image1 = frame
            cv2.imwrite(name, self.image1)
        elif index == 1:
            self.image2 = frame
            cv2.imwrite(name, self.image2)
        elif index == 2:
            self.image3 = frame
            cv2.imwrite(name, self.image3)
        elif index == 3:
            self.image4 = frame
            cv2.imwrite(name, self.image4)
        elif index == 4:
            self.image5 = frame
            cv2.imwrite(name, self.image5)
        elif index == 5:
            self.image6 = frame
            cv2.imwrite(name, self.image6)

        # calling newly created image
        self.root.ids[id_name].source = directory + "/" + name

    def change_texture(self, index, root):
        expressions = ["NEUTRAL", "SMILE", "RAISE EYEBROWS", "FROWN"]
        root.ids['label1'].text = expressions[index]

    def close_test(self):
        if self.capture is not None:
            self.clock.cancel()
            self.capture.release()
        screen_manager.current = "ExamPage"

    def show_mp_results(self, bob):
        bob.ids[
            'mp_image1'].source = r'../temp_images/MP_img1.jpg'
        bob.ids[
            'mp_image2'].source = r'../temp_images/MP_img2.jpg'
        bob.ids[
            'mp_image3'].source = r'../temp_images/MP_img3.jpg'
        bob.ids[
            'mp_image4'].source = r'../temp_images/MP_img4.jpg'

        bob.ids['LeftAlt'].text = "Avg. Left Alteration: " + str(self.results[1])
        bob.ids['RightAlt'].text = "Avg. Right Alteration: " + str(self.results[0])
        bob.ids['nrValue'].text = "nR-Value: " + str(self.results[2])
        bob.ids['nu_rValue'].text = "nu_r-Value: " + str(self.results[3])

    def MP_Method(self):

        # MediaPipe
        image_test = MediaPipe_Method(self.refs)

        img1_results, img1_out = image_test.mp_process(self.image1)
        img2_results, img2_out = image_test.mp_process(self.image2)
        img3_results, img3_out = image_test.mp_process(self.image3)
        img4_results, img4_out = image_test.mp_process(self.image4)

        self.mp_result1 = img1_out
        self.mp_result2 = img2_out
        self.mp_result3 = img3_out
        self.mp_result4 = img4_out

        cv2.imwrite("MP_img1.jpg", self.mp_result1)
        cv2.imwrite("MP_img2.jpg", self.mp_result2)
        cv2.imwrite("MP_img3.jpg", self.mp_result3)
        cv2.imwrite("MP_img4.jpg", self.mp_result4)

        # GCM
        patient = Geometric_Computation([img1_results, img2_results])
        # patient.pop_refs()
        patient.show_dicts()
        patient.factor_dicts()
        patient.all_diffs()
        patient.show_results()

        self.results = patient.results

    def refresh_test(self, root):
        root.ids["label1"].text = "START"
        root.ids["web_feed"].source = r'../Kivy Logo Images/TestLiveStream.png'
        root.ids["image1"].source = r'../Kivy Logo Images/4.png'
        root.ids["image2"].source = r'../Kivy Logo Images/5.png'
        root.ids["image3"].source = r'../Kivy Logo Images/6.png'
        root.ids["image4"].source = r'../Kivy Logo Images/7.png'

    def delete_temps(self):
        for filename in os.listdir(r'../temp_images'):
            try:
                os.remove('*')
            except OSError:
                pass
        return "TEMP PHOTOS REMOVED"


class LoadingScreen(Screen):
    pass


class AboutPage(Screen):
    pass


class StartExam(Screen):
    pass


class MyApp(MDApp):
    global screen_manager
    screen_manager = ScreenManager()

    def build(self):
        # Set App Title
        self.title = "NeuroVA"
        # Set App theme
        self.theme_cls.primary_palette = 'Green'
        self.theme_cls.primary_hue = '100'

        # image for life feed
        self.MP = MPFaceMesh()

        # self.web_feed = Image()
        screen_manager.add_widget(Builder.load_file("loadingScreen.kv"))
        screen_manager.add_widget(Builder.load_file("mainScreen.kv"))
        screen_manager.add_widget(Builder.load_file("aboutScreen.kv"))
        screen_manager.add_widget(Builder.load_file("startExam.kv"))
        screen_manager.add_widget(Builder.load_file("mpFaceMesh.kv"))
        screen_manager.add_widget(Builder.load_file("profileScreen.kv"))
        screen_manager.add_widget(Builder.load_file("results_page.kv"))

        return screen_manager

    def for_now(self):
        print('test')

    def on_start(self):
        Clock.schedule_once(self.change_screen, 10)

    def change_screen(self, *args):
        screen_manager.current = "MainScreen"

    def go_back_screen_once(self, name):
        screen_manager.current = name

    def on_tab_switch(self, *args):
        # instance_tabs, instance_tab, instance_tab_label, tab_text
        MyApp.go_back_screen_once(None, args[3].replace(' ', ''))

if __name__ == '__main__':
    MyApp().run()
