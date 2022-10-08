from Code.face_mesh_mediapipe import MediaPipe_Method
from GCM.geometric_computation import Geometric_Computation
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
import cv2
from kivymd.uix.snackbar import Snackbar
from kivy.metrics import dp
from kivymd.uix.datatables import MDDataTable
import os
import mediapipe as mp

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

    def load_video(self, *args):
        y = 150
        x = 175
        h = 240
        w = 300
        # read frame from opencv
        if self.capture is None:
            self.clock.cancel()
            self.capture.release()

        ret, frame = self.capture.read()
        self.frame = frame[y:y + h, x:x + w]

        # Flip horizontal and convert image to texture
        buffer = cv2.flip(self.frame, 0).tobytes()
        texture = Texture.create(size=(self.frame.shape[1], self.frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.root.ids['web_feed'].texture = texture

    def take_picture(self, index):
        # ensuring directory for later deletion
        directory = r'C:\Users\Alex Estrada\PycharmProjects\FaceMeshSeniorDesign\temp_images'
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
        # elif index == 4:
        #     self.image5 = frame
        #     cv2.imwrite(name, self.image5)
        # elif index == 5:
        #     self.image6 = frame
        #     cv2.imwrite(name, self.image6)

        # calling newly created image
        self.root.ids[id_name].source = directory + "\\" + name

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
            'mp_image1'].source = r'C:\Users\Alex Estrada\PycharmProjects\FaceMeshSeniorDesign\temp_images\MP_img1.jpg'
        bob.ids[
            'mp_image2'].source = r'C:\Users\Alex Estrada\PycharmProjects\FaceMeshSeniorDesign\temp_images\MP_img2.jpg'
        bob.ids[
            'mp_image3'].source = r'C:\Users\Alex Estrada\PycharmProjects\FaceMeshSeniorDesign\temp_images\MP_img3.jpg'
        bob.ids[
            'mp_image4'].source = r'C:\Users\Alex Estrada\PycharmProjects\FaceMeshSeniorDesign\temp_images\MP_img4.jpg'

        # bob.ids['LeftAlt'].text = "Avg. Left Alteration: " + str(self.results[1])
        # bob.ids['RightAlt'].text = "Avg. Right Alteration: " + str(self.results[0])
        # bob.ids['nrValue'].text = "nR-Value: " + str(self.results[2])
        # bob.ids['nu_rValue'].text = "nu_r-Value: " + str(self.results[3])

    def MP_Method(self):
        pre_images = [self.image1, self.image2, self.image3, self.image4]
        post_images = []

        for i in pre_images:
            if i is not None:
                post_images.append(i)

        if len(post_images) != 4:
            print("NOT ALL EXPRESSIONS ARE BEING TESTED")

        # MediaPipe
        image_test = MediaPipe_Method(self.refs, post_images)
        original_dicts, mirrored_dicts, mp_imgs, mp_mirrored_imgs = image_test.mp_run("", 0)

        self.mp_result1 = mp_imgs[0]
        self.mp_result2 = mp_imgs[1]
        self.mp_result3 = mp_imgs[2]
        self.mp_result4 = mp_imgs[3]

        cv2.imwrite("MP_img1.jpg", self.mp_result1)
        cv2.imwrite("MP_img2.jpg", self.mp_result2)
        cv2.imwrite("MP_img3.jpg", self.mp_result3)
        cv2.imwrite("MP_img4.jpg", self.mp_result4)

        # GCM
        patient = Geometric_Computation(original_dicts)
        mirrored_patient = Geometric_Computation(mirrored_dicts)

        original_mirrored_dist, all_avg, upper_lower_split_avg = patient.get_icp_distances(
            mirrored_patient.norm_array_dicts)

        for idx, i in enumerate(original_mirrored_dist):
            print("----------------------------------------ALTERNATE MODEL "
                  "RESULTS-----------------------------------------")
            print("IMG", idx + 1, ":")
            # print("All original-to-mirrored distances:", i)
            print("Upper Asymmetry Score:", upper_lower_split_avg[idx][0] * 10000)
            print("Lower Asymmetry Score:", upper_lower_split_avg[idx][1] * 10000)
            print("Weighted Average:", all_avg[idx] * 10000)

        # patient.GCM1()
        patient.GCM2()

    def refresh_test(self, root):
        root.ids["label1"].text = "START"
        root.ids["web_feed"].source = r'C:/Users\Alex Estrada/PycharmProjects/FaceMeshSeniorDesign/Kivy Logo ' \
                                      r'Images/TestLiveStream.png '
        root.ids["image1"].source = r'C:/Users\Alex Estrada/PycharmProjects/FaceMeshSeniorDesign/Kivy Logo ' \
                                    r'Images/4.png '
        root.ids["image2"].source = r'C:/Users\Alex Estrada/PycharmProjects/FaceMeshSeniorDesign/Kivy Logo ' \
                                    r'Images/5.png '
        root.ids["image3"].source = r'C:/Users\Alex Estrada/PycharmProjects/FaceMeshSeniorDesign/Kivy Logo ' \
                                    r'Images/6.png '
        root.ids["image4"].source = r'C:/Users\Alex Estrada/PycharmProjects/FaceMeshSeniorDesign/Kivy Logo ' \
                                    r'Images/7.png '

    def delete_temps(self):
        for filename in os.listdir(r'C:\Users\Alex Estrada\PycharmProjects\FaceMeshSeniorDesign\temp_images'):
            # print(filename)
            os.remove(filename)
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

        return screen_manager

    def on_start(self):
        # 15 sec delay
        # screen_manager.current = 'MPFaceMesh'
        Clock.schedule_once(self.change_screen, 1)

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
