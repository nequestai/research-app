"""
Concrete MediaPipe module


# Copyright (c) 2022-Current Alex Estrada <aestradab@ucdavis.edu>
"""

import mediapipe as mp
import pandas as pd
import cv2


def flip_img(img):
    return cv2.flip(img, 1)


class MediaPipe_Method:

    mouth_index = [61, 291, 76, 306, 62, 292, 78, 308, 191, 415, 80, 310, 95, 324, 88, 318, 184, 408,
                   74, 304, 183, 407, 42, 272, 96, 325, 89, 319, 77, 307, 90, 320, 73, 303, 72, 302, 41,
                   271, 38, 268, 81, 311, 82, 312, 178, 402, 87, 317, 179, 403, 86, 316, 180, 404, 85, 315,
                   57, 287, 185, 409, 40, 270, 39, 269, 37, 267, 146, 375, 91, 321, 181, 405, 84, 314]
    chin_cheeks = [83, 313, 201, 421, 208, 428, 171, 396, 182, 406, 194, 418, 32, 262, 140, 369, 106, 335,
                   204, 424, 211, 431, 43, 273, 202, 422, 210, 430, 212, 432, 214, 434, 135, 364, 167, 393,
                   165, 391, 92, 322, 186, 410, 186, 410, 203, 423, 206, 426, 216, 436, 36, 266, 205, 425,
                   207, 427, 187, 411, 192, 416, 50, 280, 142, 371, 209, 429, 129, 358, 213, 433]
    eye_index = [225, 445, 224, 444, 223, 443, 222, 442, 221, 441, 33, 263, 246, 466, 161, 388, 160, 387,
                 159, 386, 158, 385, 157, 384, 173, 398, 133, 362, 7, 249, 163, 390, 144, 373, 145, 374,
                 153, 380, 154, 381, 155, 382]
    eyebrow_index = [70, 300, 63, 293, 105, 334, 66, 296, 107, 336, 46, 276,
                     53, 283, 52, 282, 65, 295, 55, 285, 189, 413]
    forehead_near_eye = [193, 417, 189, 413, 122, 351, 245, 465, 188, 412, 196, 419, 3, 248, 174, 399, 244, 464,
                         128, 357, 114, 343, 243, 463, 233, 453, 108, 337, 69, 299, 104, 333, 68, 298, 156, 383,
                         124, 353, 113, 342, 190, 414, 56, 286, 28, 258, 27, 257, 29, 259, 30, 260, 247, 467,
                         130, 359, 25, 255, 110, 339, 24, 254, 23, 253, 22, 252, 26, 256, 112, 341, 232, 452,
                         121, 350, 231, 451, 230, 450, 229, 449, 228, 448, 31, 261, 226, 446, 35, 265, 143, 372,
                         111, 340, 117, 346, 118, 347, 119, 348, 120, 349, 47, 277, 217, 437]

    # center_index = [0, 2, 1, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 94, 151, 164, 168,
    #                 175, 197, 195, 199, 200]
    center_index = [6]
    upper = eyebrow_index + eye_index + forehead_near_eye
    lower = mouth_index + chin_cheeks

    def __init__(self, references=None, imgs=None, upper=upper, lower=lower, center=center_index):
        self.refs = references
        self.imgs = imgs

        self.upper = upper
        self.lower = lower
        self.center = center

        self.all_index = self.upper + self.lower + self.refs + self.center

    def mp_run(self, name=None, save=True):
        raw_imgs = self.imgs
        mirrored_imgs = []
        for img in raw_imgs:
            mirrored_imgs.append(cv2.flip(img, 1))

        img_dicts = []
        edited_imgs = []
        mirrored_dicts = []
        edited_mirrored_imgs = []
        all_landmarks_originals = []
        all_landmarks_mirrored_plural = []

        for idx, img in enumerate(raw_imgs):
            if hasattr(img, 'shape'):
                img_dict, edited_img, all_landmarks_original = self.mp_process(img)
                mirrored_dict, edited_mirrored_img, all_landmarks_mirrored = self.mp_process(mirrored_imgs[idx])
                # appending dictionary list
                img_dicts.append(img_dict)
                mirrored_dicts.append(mirrored_dict)
                # appending images
                edited_imgs.append(edited_img)
                edited_mirrored_imgs.append(edited_mirrored_img)
                # appending all landmarks
                all_landmarks_originals.append(all_landmarks_original)
                all_landmarks_mirrored_plural.append(all_landmarks_mirrored)

        if save:
            path = "./data/" + name + ".csv"

            dfs_temp = []
            dfs_temp2 = []
            for i in all_landmarks_originals:
                df = pd.DataFrame.from_dict(i, orient='index', columns=['X', 'Y'])
                dfs_temp.append(df)
            for i in all_landmarks_mirrored_plural:
                df = pd.DataFrame.from_dict(i, orient='index', columns=["I", "J"])
                dfs_temp2.append(df)

            result_df = pd.concat(dfs_temp, axis=1)     # original
            result_df2 = pd.concat(dfs_temp2, axis=1)   # mirrored
            final_df = pd.concat([result_df, result_df2], axis=1)

            final_df.to_csv(path)
        return img_dicts, mirrored_dicts, edited_imgs, edited_mirrored_imgs

    def mp_process(self, img):
        index_list = self.all_index
        mp_face_mesh = mp.solutions.face_mesh
        img_dict = {}
        all_landmarks = {}
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
                min_detection_confidence=0.5) as face_mesh:
            height, width, val = img.shape
            edited_img = img.copy()
            # covert BGR to RGB before processing
            results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for face_landmarks in results.multi_face_landmarks:

            for k in range(0, 468):
                x1 = int(face_landmarks.landmark[k].x * width)
                y1 = int(face_landmarks.landmark[k].y * height)
                all_landmarks[k] = [x1, y1]

            for j in index_list:
                x = int(face_landmarks.landmark[j].x * width)
                y = int(face_landmarks.landmark[j].y * height)
                img_dict[j] = [x, y]
                cv2.circle(edited_img, (x, y), 3, (255, 0, 0), 3)  # BRG

        # cv2.imshow("IMG", edited_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return img_dict, edited_img, all_landmarks
