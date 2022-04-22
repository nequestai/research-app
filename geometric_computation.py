"""
Concrete MediaPipe module


# Copyright (c) 2022-Current Alex Estrada <aestradab@ucdavis.edu>
"""

import matplotlib.pyplot as plt
from face_mesh_mediapipe import MediaPipe_Method
import math
import numpy as np
from icp import icp


class Geometric_Computation(MediaPipe_Method):

    def __init__(self, dicts=None, upper=[], lower=[], center=[]):
        super(Geometric_Computation, self).__init__(upper, lower, center)
        self.dicts = dicts
        self.refs = [127, 356]

        # for sake of old code
        self.norm_array_dict1 = None
        self.norm_array_dict2 = None

        # updated
        # self.norm_array_dict = None
        self.results = None

        # for new computations
        self.norm_array_dicts = []
        self.mid = []
        self.upper_splits = []
        self.lower_splits = []

        self.pop_refs(self.refs)

        # results GCM
        self.upper_diffs_GCM = None
        self.lower_diffs_GCM = None
        self.all_diffs_GCM = None
        self.average_uppers_GCM = None
        self.average_lowers_GCM = None
        # results alternate model
        self.original_mirrored_distances_alternate = None
        self.upper_lower_splits_alternate = None
        self.all_average_alternate = None

    def pop_refs(self, refs):
        ref = []
        mid = []
        for i in self.dicts:
            mid_temp = []
            ref.append([i.pop(refs[0]), i.pop(refs[1])])
            for mid_point in self.center:
                mid_temp.append(i.pop(mid_point))
            mid.append(mid_temp)
        self.refs = ref
        self.mid = mid

        self.show_dicts()
        self.normalize_dicts()

    def show_dicts(self):
        for idx, dict in enumerate(self.dicts):
            print("IMG", idx, "|", len(dict.items()), "Landmarks |", dict.items())

    def factor_dicts(self):
        factors = []
        for ref in self.refs:
            dif = np.subtract(np.array(ref[0]), np.array(ref[1]))
            factors.append(math.sqrt(dif[0] ** 2 + dif[1] ** 2))
        print("Factors of Images:", factors)
        return factors

    def sagittalize(self):
        out = []
        for i in self.mid:
            x = []
            y = []
            for j in i:
                x.append(j[0])
                y.append(j[1])
            out.append([np.mean(x), np.mean(y)])
        print("Center Points:", out)
        return out

    def normalize_dicts(self):
        # get normalizing factors from existing images
        factors = Geometric_Computation.factor_dicts(self)
        # getting sagittal line
        sagittal = Geometric_Computation.sagittalize(self)

        for idx, mp_dict in enumerate(self.dicts):
            np_array_dict = np.array(list(mp_dict.values()))
            self.norm_array_dicts.append(np_array_dict)
            self.mid[idx] = np.array(sagittal[idx])

        Geometric_Computation.mid_norm_plot(self, 0)  # Default is true to plot

        for idx, i in enumerate(self.norm_array_dicts):
            self.norm_array_dicts[idx] = i / factors[idx]

        print("Sagittal line reference computed:", sagittal)
        print("Normalization of", len(self.norm_array_dicts), "dictionaries is complete.")

    def mid_norm_plot(self, plot=True):
        for idx, norm_dict in enumerate(self.norm_array_dicts):
            self.norm_array_dicts[idx] = self.norm_array_dicts[idx] - self.mid[idx]
        # print("Scaling with respect to sagittal line complete")
        if plot:
            for i in self.norm_array_dicts:
                x = []
                y = []
                for j in i:
                    x.append(j[0])
                    y.append(j[1])
                plt.plot(x, y, 'o')
                plt.gca().invert_yaxis()
                plt.show()

    def regional_split(self):
        upper_num = len(self.eyebrow_index + self.eye_index)
        lower_num = len(self.mouth_index)

        for i in self.norm_array_dicts:
            self.upper_splits.append(i[0:upper_num])  # check for indexing. possibly should be 0:upper - 2 (-1?)
            self.lower_splits.append(i[upper_num:upper_num + lower_num])

        print(len(self.upper_splits), "upper & ", len(self.lower_splits), " lower splits successful")
        print(len(self.upper_splits[0]), "upper & ", len(self.lower_splits[0]), "lower landmarks")

    def upper_diffs(self):
        base = self.upper_splits.pop(0)
        upper_euclidean_distances = []
        for idx, i in enumerate(self.upper_splits):
            upper_euclidean_distances.append(np.linalg.norm(base - i, axis=1))

        # saving results of upper differences
        self.upper_diffs_GCM = upper_euclidean_distances

    def lower_diffs(self):
        base = self.lower_splits.pop(0)
        lower_euclidean_distances = []
        for idx, i in enumerate(self.lower_splits):
            lower_euclidean_distances.append(np.linalg.norm(base - i, axis=1))

        # saving results of lower differences
        self.lower_diffs_GCM = lower_euclidean_distances

    def all_diffs(self):
        base = self.norm_array_dicts.pop(0)
        all_euclidean_distances = []
        for idx, i in enumerate(self.norm_array_dicts):
            all_euclidean_distances.append(np.linalg.norm(base - i, axis=1))

        # saving results of lower differences
        self.all_diffs_GCM = all_euclidean_distances

    def total_diffs(self):
        Geometric_Computation.regional_split(self)
        Geometric_Computation.upper_diffs(self)
        Geometric_Computation.lower_diffs(self)
        Geometric_Computation.all_diffs(self)
        print("Total distances across images computed")

    def GCM1(self):
        print('--------------------------------------------STARTING GCM1----------------------------------------------')
        Geometric_Computation.total_diffs(self)
        upper = self.upper_diffs_GCM
        lower = self.lower_diffs_GCM
        both = self.all_diffs_GCM

        upper_dict = {'dr': [], 'dl': []}
        lower_dict = {'dr': [], 'dl': []}
        all_dict = {'dr': [], 'dl': []}

        # splitting between dr and dl
        for idx, i in enumerate([upper, lower, both]):
            if idx == 0:
                for j in i:
                    upper_dict['dr'].append(np.array(j[::2]))
                    upper_dict['dl'].append(np.array(j[1::2]))
            if idx == 1:
                for j in i:
                    lower_dict['dr'].append(np.array(j[::2]))
                    lower_dict['dl'].append(np.array(j[1::2]))
            if idx == 2:
                for j in i:
                    all_dict['dr'].append(np.array(j[::2]))
                    all_dict['dl'].append(np.array(j[1::2]))

        # now for the r and avg_r values
        r_upper = []
        r_lower = []
        r_all = []
        for i in range(0, len(upper_dict['dr'])):   # getting length of images to do GCM
            # print("UPPER DR", len(upper_dict['dr'][i]), upper_dict['dr'][i])
            # print("UPPER DL", len(upper_dict['dl'][i]), upper_dict['dl'][i])
            r_upper.append(abs(1 - upper_dict['dl'][i]/upper_dict['dr'][i]))
            r_lower.append(abs(1 - lower_dict['dl'][i]/lower_dict['dr'][i]))
        for i in range(0, len(all_dict['dr'])):
            r_all.append(abs(1 - all_dict['dl'][i] / all_dict['dr'][i]))

        avg_r_upper = []
        avg_r_lower = []
        avg_r_all = []
        for i in range(0, len(r_upper)):
            avg_r_upper.append(np.sum(r_upper[i])/(len(r_upper[0] / 2)))
            avg_r_lower.append(np.sum(r_lower[i]) / (len(r_lower[0] / 2)))
        for i in range(0, len(r_all)):
            avg_r_all.append(np.sum(r_all[i]) / (len(r_all[0] / 2)))

        print("UPPER:", avg_r_upper)
        print("LOWER:", avg_r_lower)
        print("Weighted Average:", avg_r_all)

    def GCM2(self):
        print('--------------------------------------------STARTING GCM2----------------------------------------------')
        Geometric_Computation.total_diffs(self)
        upper = self.upper_diffs_GCM
        lower = self.lower_diffs_GCM
        both = self.all_diffs_GCM

        upper_dict = {'dr': [], 'dl': []}
        lower_dict = {'dr': [], 'dl': []}
        all_dict = {'dr': [], 'dl': []}

        # splitting between dr and dl
        for idx, i in enumerate([upper, lower, both]):
            if idx == 0:
                for j in i:
                    upper_dict['dr'].append(np.array(j[::2]))
                    upper_dict['dl'].append(np.array(j[1::2]))
            if idx == 1:
                for j in i:
                    lower_dict['dr'].append(np.array(j[::2]))
                    lower_dict['dl'].append(np.array(j[1::2]))
            if idx == 2:
                for j in i:
                    all_dict['dr'].append(np.array(j[::2]))
                    all_dict['dl'].append(np.array(j[1::2]))

        # now for the r and avg_r values
        r_upper = []
        r_lower = []
        r_all = []
        for i in range(0, len(upper_dict['dr'])):  # getting length of images to do GCM
            # print("UPPER DR", len(upper_dict['dr'][i]), upper_dict['dr'][i])
            # print("UPPER DL", len(upper_dict['dl'][i]), upper_dict['dl'][i])
            r_upper.append(abs(upper_dict['dl'][i] - upper_dict['dr'][i]))
            r_lower.append(abs(lower_dict['dl'][i] - lower_dict['dr'][i]))
        for i in range(0, len(all_dict['dr'])):
            r_all.append(abs(all_dict['dl'][i] - all_dict['dr'][i]))

        avg_r_upper = []
        avg_r_lower = []
        avg_r_all = []
        for i in range(0, len(r_upper)):
            avg_r_upper.append(np.sum(r_upper[i]) / (len(r_upper[0] / 2)))
            avg_r_lower.append(np.sum(r_lower[i]) / (len(r_lower[0] / 2)))
        for i in range(0, len(r_all)):
            avg_r_all.append(np.sum(r_all[i]) / (len(r_all[0] / 2)))

        print("UPPER:", avg_r_upper)
        print("LOWER:", avg_r_lower)
        print("Weighted Average:", avg_r_all)

    @staticmethod
    def compute_icp(reference_points, points):
        transformation_history, aligned_points = icp(reference_points,
                                                     points,
                                                     verbose=False)
        # show results
        # plt.plot(reference_points[:, 0], reference_points[:, 1], 'rx', label='original points')
        # plt.plot(points[:, 0], points[:, 1], 'b1', label='mirrored points')
        # plt.plot(aligned_points[:, 0], aligned_points[:, 1], 'g+', label='aligned points')
        # plt.legend()
        # plt.gca().invert_yaxis()
        # plt.show()
        # print("ICP Complete")
        return reference_points, aligned_points

    def get_icp_distances(self, points):

        reference_points = self.norm_array_dicts

        distances = []
        for idx, array in enumerate(reference_points):
            out, aligned = Geometric_Computation.compute_icp(array, points[idx])
            distances.append(np.linalg.norm(array - aligned, axis=1))

        upper_num = len(self.eyebrow_index + self.eye_index)

        average_all = []
        upper_lower = []

        for i in distances:
            high = np.mean(i[0:upper_num - 1])
            low = np.mean(i[upper_num:])

            upper_lower.append([high, low])
            average_all.append(np.mean(i))

        self.original_mirrored_distances_alternate = distances
        self.all_average_alternate = average_all
        self.upper_lower_splits_alternate = upper_lower

        return distances, average_all, upper_lower

    def save_results(self):
        print("Geometric Computation for landmarks:", self.all_index,
              "\nReference Landmarks:", self.refs,
              "\nAvg. Left Alteration:", np.mean(self.results[1]),
              "\nRight Alteration:", np.mean(self.results[0]),
              "\nR-Value:", np.mean(self.results[2]),
              "\nu_r Value:", self.results[3])
