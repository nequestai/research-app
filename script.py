"""
NeuroVA Script


# Copyright (c) 2022-Current Alex Estrada <aestradab@ucdavis.edu>
"""

from Code.face_mesh_mediapipe import MediaPipe_Method
from GCM.geometric_computation import Geometric_Computation
import numpy as np
import cv2
import matplotlib.pyplot as plt


# loading files
# names: Alex, Gigi, Lauren, Kelsey, Love, Nicole, Valeria, Jordan, Koush, Javier, Makenna, Lanie, Michael, Matthew,
# Trevor
name = 'Alex'
expressions = ['Neutral', 'Smile', 'Wow', 'Frown']
img_to_test = []
for i in expressions:
    path = './Images/' + name + i + '.jpg'
    img_to_test.append(cv2.imread(path))

# refs
refs_top_bot = [10, 152]
refs_sides = [127, 356]

# call MediaPipe, initialize indexes, reference points, and all required images
image_test = MediaPipe_Method(refs_sides, img_to_test)
original_dicts, mirrored_dicts, mp_imgs, mp_mirrored_imgs = image_test.mp_run("", 0)    # name, to save CSV=True
# original dicts: ----- dictionaries of the landmarks of original photos
# mirrored dicts: ----- dictionaries of the landmarks of mirrored images
# mp_imgs: ------------ edited image of the original photo with landmarks
# mp_mirrored_imgs: --- edited image of the mirrored photo with landmarks

# scale_percent = 40  # percent of original size
# mp_resized = []
# for i in mp_imgs:
#     width = int(i.shape[1] * scale_percent / 100)
#     height = int(i.shape[0] * scale_percent / 100)
#     dim = (width, height)
#
#     # resize image
#     resized = cv2.resize(i, dim, interpolation=cv2.INTER_AREA)
#     mp_resized.append(resized)
# cv2.imshow("R0", mp_resized[0])
# cv2.imshow("R1", mp_resized[1])
# cv2.imshow("R2", mp_resized[2])
# cv2.imshow("R3", mp_resized[3])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# initiating GCM with original and mirrored dictionaries. References and sagittal points popped upon initialization.
patient = Geometric_Computation(original_dicts)
mirrored_patient = Geometric_Computation(mirrored_dicts)

# alternate model
original_mirrored_dist, all_avg, upper_lower_split_avg = patient.get_icp_distances(mirrored_patient.norm_array_dicts)

for idx, i in enumerate(original_mirrored_dist):
    print("----------------------------------------ALTERNATE MODEL RESULTS-----------------------------------------")
    print("IMG", idx+1, ":")
    # print("All original-to-mirrored distances:", i)
    print("Upper Asymmetry Score:", upper_lower_split_avg[idx][0]*10000)
    print("Lower Asymmetry Score:", upper_lower_split_avg[idx][1]*10000)
    print("Weighted Average:", all_avg[idx]*10000)

# patient.GCM1()
patient.GCM2()
