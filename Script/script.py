"""
NeuroVA Script


# Copyright (c) 2022-Current Alex Estrada <aestradab@ucdavis.edu>
"""

from face_mesh_mediapipe import MediaPipe_Method
from geometric_computation import Geometric_Computation
import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt


# loading files
# names: Alex, Gigi, Lauren, Kelsey, Love, Nicole, Valeria, Jordan, Koush, JavierP, Makenna, Lanie, Michael, Matthew,
# Trevor, Nancy, Marianne, Jayna, Elise, Cara, Cindee, Jake, Adam, (Maureen-lost), Katie, Leslie, Kevin, Meg, Lee,
# Megan, Jay, Barbara, Brooke, MrsBond, Gretchen, Andie-, Steve , Karen , Grace , Iris , ###

# Abnormal
# Alex2, Kelsey2, Lauren2

# Clinic
# DA2_

# DATASETS
names_smile_intensity = ['Alex', 'Gen', 'Robby', 'Robby2', 'Keith', 'Keith2', 'Bhargavi', 'Reza', 'Janice',
                         'MsBond', 'Valeria', 'Karla', 'Kelsey', 'Lauren', 'Ben', 'Jessica', 'Alex2', 'Gen2']
# Alex2 and Gen2 = 15 degree tilt of camera. Exact same photo as original
# Keith2 changes hat. Robby2 changes glasses

name = 'Bhargavi'
expressions = ['Neutral', 'Smile', 'Wow', 'Frown']
expressions_smile_level_test = ['Smile_0', 'Smile_1', 'Smile_2', 'Smile_3']

expressions_smile_level_test_test = ['Smile_0', 'Smile_1', 'Smile_1', 'Smile_1']
temp_img_to_test = []
for i in expressions_smile_level_test_test:
# for i in expressions:
    # path = '../Images/' + name + i + '.jpg'
    path = '../Images/' + name + i + '.jpg'
    temp_img_to_test.append(cv2.imread(path))

# pre-processing of images
scale = 800
img_to_test = []
for i in temp_img_to_test:
    width = int(i.shape[1] * scale / 100)
    height = int(i.shape[0] * scale / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(i, dim, interpolation=cv2.INTER_AREA)
    img_to_test.append(resized)

# refs
refs_top_bot = [10, 152]
refs_sides = [127, 356]

# call MediaPipe, initialize indexes, reference points, and all required images
image_test = MediaPipe_Method(refs_sides, img_to_test)
original_dicts, mirrored_dicts, mp_imgs, mp_mirrored_imgs = image_test.mp_run(name, 0)    # name, to save CSV=True
# original dicts: ----- dictionaries of the landmarks of original photos
# mirrored dicts: ----- dictionaries of the landmarks of mirrored images
# mp_imgs: ------------ edited image of the original photo with landmarks
# mp_mirrored_imgs: --- edited image of the mirrored photo with landmarks

scale_percent = 30  # percent of original size
mp_resized = []
for i in mp_imgs:
    width = int(i.shape[1] * scale_percent / 100)
    height = int(i.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(i, dim, interpolation=cv2.INTER_AREA)
    mp_resized.append(resized)
cv2.imshow("R0", mp_resized[0])
cv2.imshow("R1", mp_resized[1])
cv2.imshow("R2", mp_resized[2])
cv2.imshow("R3", mp_resized[3])
cv2.waitKey(0)
cv2.destroyAllWindows()

# initiating GCM with original and mirrored dictionaries. References and sagittal points popped upon initialization.
patient = Geometric_Computation(original_dicts)
mirrored_patient = Geometric_Computation(mirrored_dicts)

# alternate model
original_mirrored_dist, all_avg, upper_lower_split_avg = patient.get_icp_distances(mirrored_patient.norm_array_dicts)

alternate_model_avg = [name]    # avg1, avg2, avg3, avg4
alternate_model_up_low = []     # up1, lo1, up1, lo1, up1, lo1, up1, lo1
for idx, i in enumerate(original_mirrored_dist):
    print("----------------------------------------ALTERNATE MODEL RESULTS-----------------------------------------")
    print("IMG", idx+1, ":")
    # print("All original-to-mirrored distances:", i)
    alternate_model_avg.append(all_avg[idx]*10000)
    alternate_model_up_low.append(upper_lower_split_avg[idx][0]*10000)
    alternate_model_up_low.append(upper_lower_split_avg[idx][1]*10000)
    print("Upper Asymmetry Score:", upper_lower_split_avg[idx][0]*10000)
    print("Lower Asymmetry Score:", upper_lower_split_avg[idx][1]*10000)
    print("Weighted Average:", all_avg[idx]*10000)

avg1_r_all, avg1_r_upper, avg1_r_lower, upper_dr, upper_dl, lower_dr, lower_dl = patient.GCM1()
avg2_r_all, avg2_r_upper, avg2_r_lower = patient.GCM2()
avg3_r_all, avg3_r_upper, avg3_r_lower, ratios = patient.GCM3()
avg3_2_r_all, avg3_2_r_upper, avg3_2_r_lower = patient.GCM3_2()

# organizing for CSV -------------------------------------------------------------------------------------------------
dl_dr = [upper_dr[0], upper_dl[0], lower_dr[0], lower_dl[0],
         upper_dr[1], upper_dl[1], lower_dr[1], lower_dl[1],
         upper_dr[2], upper_dl[2], lower_dr[2], lower_dl[2]]

temp = [avg1_r_upper[0], avg1_r_lower[0], avg1_r_upper[1], avg1_r_lower[1], avg1_r_upper[2], avg1_r_lower[2]]
temp2 = [avg2_r_upper[0], avg2_r_lower[0], avg2_r_upper[1], avg2_r_lower[1], avg2_r_upper[2], avg2_r_lower[2]]
temp3 = [avg3_r_upper[0], avg3_r_lower[0], avg3_r_upper[1], avg3_r_lower[1], avg3_r_upper[2], avg3_r_lower[2]]
temp_ratios = [ratios[0], ratios[1], ratios[2]]

gcm1 = [name] + dl_dr + [name] + avg1_r_all + temp  # this has the dl_dr which is not GCM1 specific
GCM1 = [name] + avg1_r_all + temp
gcm2 = [name] + avg2_r_all + temp2
gcm3 = [name] + avg3_r_all.tolist() + temp3

gcm1_ = [name] + avg1_r_all
gcm2_ = [name] + avg2_r_all
gcm3_ = [name] + avg3_r_all.tolist()
gcm3_2 = [name] + avg3_2_r_all.tolist()
# final output for CSV
out = alternate_model_avg + alternate_model_up_low + gcm1 + gcm2
out3 = alternate_model_avg + gcm1_ + gcm2_ + gcm3_ + gcm3_2 + temp_ratios   # used for smile intensity testing

# saving comments and date to csv -----------------------------------------------------------------------------------
# directory = r'./data/all_results_extra_landmarks_patients.csv'
directory = r'./data/Smile_Intensity_Data_9_19.csv'
# directory = r'./data/normalized_GCM.csv'

# with open(directory, 'a', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(out3)
