from face_mesh_mediapipe import MediaPipe_Method
from geometric_computation import Geometric_Computation
from kivymd.app import MDApp
from kivymd.uix.menu import MDDropdownMenu
from kivy.properties import ObjectProperty
from kivy.lang import Builder
from kivymd.uix.screen import Screen
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivymd.uix.tab import MDTabsBase, MDTabs
from kivy.uix.screenmanager import ScreenManager
from kivy.core.window import Window
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.gridlayout import MDGridLayout
from kivymd.uix.floatlayout import MDFloatLayout
from kivy.uix.scrollview import ScrollView
from kivymd.uix.list import MDList
from kivymd.uix.button import MDRaisedButton, MDRectangleFlatButton
from kivymd.uix.label import MDLabel
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivymd.uix.behaviors import FakeRectangularElevationBehavior
from kivy.clock import Clock
from kivy.uix.widget import Widget
from kivy.resources import resource_add_path
from kivymd.uix.dialog import MDDialog
import cv2
from kivymd.uix.snackbar import Snackbar
from kivy.metrics import dp
from kivymd.uix.datatables import MDDataTable
import os
import math
import mediapipe as mp
import csv
from kivy.config import Config
import pandas as pd
Config.set('graphics', 'resizable', True)
Config.set('graphics', 'width', '750')
Config.set('graphics', 'height', '1200')
Config.write()

Window.size = (90 * 5, 160 * 6)  # remove this for deployment

resource_add_path('.')

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


class MainScreen(Screen):
    pass


class UserProfile(Screen):
    pass


class NavBar(FakeRectangularElevationBehavior, MDFloatLayout):
    pass


class ExamPage(Screen, MDTabsBase, MDFloatLayout, MDDialog):
    pass


class Tab(MDTabsBase):
    pass


class DataWindow(Screen):
    pass


class MPFaceMesh(Screen, Image, MDBoxLayout):

    def __init__(self, **kw):
        super().__init__(**kw)
        self.profile_id = None
        self.profile_date = None
        self.profile_comments = None

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
        self.capture = cv2.VideoCapture(1)
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

        box = self.detect_face()
        ret, frame = self.capture.read()

        if box is not None:
            start, end = self.get_face_box(frame, detection=box)
            if start is not None and end is not None:
                sx, sy = start
                ex, ey = end

                self.frame = frame[sy-20:ey, sx:ex]
                # Flip horizontal and convert image to texture
                buffer = cv2.flip(self.frame, 0).tobytes()
                texture = Texture.create(size=(self.frame.shape[1], self.frame.shape[0]), colorfmt='bgr')
                texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
                self.root.ids['web_feed'].texture = texture
        else:
            print('face out of frame')

    def take_picture(self, index):
        # ensuring directory for later deletion
        directory = r'./temp_images'

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
            cv2.imwrite(directory + "/" + name, self.image1)
        elif index == 1:
            self.image2 = frame
            cv2.imwrite(directory + "/" + name, self.image2)
        elif index == 2:
            self.image3 = frame
            cv2.imwrite(directory + "/" + name, self.image3)
        elif index == 3:
            self.image4 = frame
            cv2.imwrite(directory + "/" + name, self.image4)

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
            'mp_image1'].source = r'./temp_images/MP_img1.jpg'
        bob.ids[
            'mp_image2'].source = r'./temp_images/MP_img2.jpg'
        bob.ids[
            'mp_image3'].source = r'./temp_images/MP_img3.jpg'
        bob.ids[
            'mp_image4'].source = r'./temp_images/MP_img4.jpg'

        bob.ids['LeftAlt'].text = "Data Collection Successful"
        bob.ids['RightAlt'].text = "ID Recorded: " + self.profile_id
        # bob.ids['nrValue'].text = "nR-Value: " + str(self.results[2])
        # bob.ids['nu_rValue'].text = "nu_r-Value: " + str(self.results[3])

    def profile_setup(self, root):
        self.profile_id = root.ids['name'].text
        self.profile_date = root.ids['date'].text
        self.profile_comments = root.ids['comment'].text

    def profile_checker(self):
        if self.profile_id is None or self.profile_id == '':
            self.profile_id = 'UNKNOWN'

    def MP_Method(self):
        directory = r'./temp_images/'
        pre_images = [self.image1, self.image2, self.image3, self.image4]
        post_images = []

        for i in pre_images:
            if i is not None:
                post_images.append(i)

        if len(post_images) != 4:
            print("NOT ALL EXPRESSIONS ARE BEING TESTED")

        # saving images to file
        # directory2 = r'./data/raw_images'
        # for idx, img in enumerate(post_images):
        #     cv2.imwrite(directory2 + "/" + self.profile_id + str(idx) + '.jpg', img)

        # saving comments and date to csv
        directory3 = r'./data/patient_id_date_info.csv'
        with open(directory3, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.profile_id, self.profile_date, self.profile_comments])

        # MediaPipe
        image_test = MediaPipe_Method(self.refs, post_images)
        original_dicts, mirrored_dicts, mp_imgs, mp_mirrored_imgs = image_test.mp_run(self.profile_id, 1)

        self.mp_result1 = mp_imgs[0]
        self.mp_result2 = mp_imgs[1]
        self.mp_result3 = mp_imgs[2]
        self.mp_result4 = mp_imgs[3]

        cv2.imwrite(directory + "MP_img1.jpg", self.mp_result1)
        cv2.imwrite(directory + "MP_img2.jpg", self.mp_result2)
        cv2.imwrite(directory + "MP_img3.jpg", self.mp_result3)
        cv2.imwrite(directory + "MP_img4.jpg", self.mp_result4)

        # GCM
        patient = Geometric_Computation(original_dicts)
        mirrored_patient = Geometric_Computation(mirrored_dicts)

        original_mirrored_dist, all_avg, upper_lower_split_avg = patient.get_icp_distances(
            mirrored_patient.norm_array_dicts)

        alternate_model_avg = []  # avg1, avg2, avg3, avg4
        alternate_model_up_low = []  # up1, lo1, up1, lo1, up1, lo1, up1, lo1
        for idx, i in enumerate(original_mirrored_dist):
            print("-------------------------------------ALTERNATE MODEL RESULTS------------------------------------")
            print("IMG", idx + 1, ":")
            # print("All original-to-mirrored distances:", i)
            alternate_model_avg.append(all_avg[idx] * 10000)
            alternate_model_up_low.append(upper_lower_split_avg[idx][0] * 10000)
            alternate_model_up_low.append(upper_lower_split_avg[idx][1] * 10000)
            print("Upper Asymmetry Score:", upper_lower_split_avg[idx][0] * 10000)
            print("Lower Asymmetry Score:", upper_lower_split_avg[idx][1] * 10000)
            print("Weighted Average:", all_avg[idx] * 10000)

        print(alternate_model_avg, alternate_model_up_low)

        avg1_r_all, avg1_r_upper, avg1_r_lower, upper_dr, upper_dl, lower_dr, lower_dl = patient.GCM1()
        avg2_r_all, avg2_r_upper, avg2_r_lower = patient.GCM2()

        # organizing for CSV ------------------------------------------------------------------------------------------
        dl_dr = [upper_dr[0], upper_dl[0], lower_dr[0], lower_dl[0],
                 upper_dr[1], upper_dl[1], lower_dr[1], lower_dl[1],
                 upper_dr[2], upper_dl[2], lower_dr[2], lower_dl[2]]
        temp = [avg1_r_upper[0], avg1_r_lower[0], avg1_r_upper[1], avg1_r_lower[1], avg1_r_upper[2], avg1_r_lower[2]]
        temp2 = [avg2_r_upper[0], avg2_r_lower[0], avg2_r_upper[1], avg2_r_lower[1], avg2_r_upper[2], avg2_r_lower[2]]
        gcm1 = [self.profile_id] + dl_dr + [self.profile_id] + avg1_r_all + temp
        gcm2 = [self.profile_id] + avg2_r_all + temp2
        # final output for CSV
        out = alternate_model_avg + alternate_model_up_low + gcm1 + gcm2
        out2 = [self.profile_id] + [patient.max] + [patient.min] + [patient.max - patient.min]

        # saving comments and date to csv -----------------------------------------------------------------------------
        directory = r'./data/all_results_extra_landmarks_patients.csv'
        with open(directory, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(out)

    def refresh_test(self, root):
        root.ids["label1"].text = "START"
        root.ids["web_feed"].source = r'./Kivy Logo Images/TestLiveStream.png'
        root.ids["image1"].source = r'./Kivy Logo Images/4.png'
        root.ids["image2"].source = r'./Kivy Logo Images/5.png'
        root.ids["image3"].source = r'./Kivy Logo Images/6.png'
        root.ids["image4"].source = r'./Kivy Logo Images/7.png'

    def delete_temps(self):
        for filename in os.listdir(r'./temp_images'):
            try:
                os.remove('*')
            except OSError:
                pass
        return print("ALL TEMP PHOTOS REMOVED")


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
        screen_manager.add_widget(Builder.load_file("instructions_page.kv"))
        screen_manager.add_widget(Builder.load_file("schedule_page.kv"))

        return screen_manager

    def on_start(self):
        # 15 sec delay
        # screen_manager.current = 'UserProfile'
        Clock.schedule_once(self.change_screen, 5)

    def change_screen(self, *args):
        screen_manager.current = "MainScreen"

    def go_back_screen_once(self, name):
        screen_manager.current = name

    def for_now(self):
        print("Alex is awesome")

    def on_tab_switch(self, *args):
        # instance_tabs, instance_tab, instance_tab_label, tab_text
        MyApp.go_back_screen_once(None, args[3].replace(' ', ''))

    # def add_datatable(self):
    #     self.data_tables = MDDataTable(
    #         size_hint=(0.9, 0.8),
    #         column_data=[
    #             ("No.", dp(30)),
    #             ("User", dp(30)),
    #             ("Password", dp(30)),
    #         ],
    #         row_data=[
    #             (
    #                 "1",
    #                 "The pitonist",
    #                 "Strong password",
    #             ),
    #             (
    #                 "2",
    #                 "The c++ lover",
    #                 "Save me!!!:)",
    #             ),
    #         ]
    #     )


if __name__ == '__main__':
    MyApp().run()
