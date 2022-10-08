"""
Concrete MediaPipe module


# Copyright (c) 2022-Current Alex Estrada <aestradab@ucdavis.edu>
"""

import matplotlib.pyplot as plt
from face_mesh_mediapipe import MediaPipe_Method
import math
import numpy as np
from icp import icp
import seaborn as sns


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

        # max movement scores for color plotting
        self.max = []
        self.min = []

        # max movement scores per two images for GCM3
        self.max_gcm3 = []
        self.min_gcm3 = []

        # results GCM
        self.upper_diffs_GCM = None
        self.lower_diffs_GCM = None
        self.all_diffs_GCM = None
        self.average_uppers_GCM = None
        self.average_lowers_GCM = None
        # results alternate model
        self.normalize_values_for_color = None
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
        upper_num = len(self.eyebrow_index + self.eye_index + self.forehead_near_eye)
        lower_num = len(self.mouth_index + self.chin_cheeks)

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
        # print(upper_euclidean_distances)

    def lower_diffs(self):
        base = self.lower_splits.pop(0)
        lower_euclidean_distances = []
        for idx, i in enumerate(self.lower_splits):
            lower_euclidean_distances.append(np.linalg.norm(base - i, axis=1))

        # saving results of lower differences
        self.lower_diffs_GCM = lower_euclidean_distances
        # print(lower_euclidean_distances)

    def all_diffs(self):
        base = self.norm_array_dicts.pop(0)
        all_euclidean_distances = []
        max_num = []
        min_num = []
        for idx, i in enumerate(self.norm_array_dicts):
            distances = np.linalg.norm(base - i, axis=1)
            all_euclidean_distances.append(distances)
            max_num.append(np.max(distances))
            min_num.append(np.min(distances))

        print('Maximum and Minimum Euclidean Distances for GCMs')
        print("--MAX", max_num)
        print("--MIN", min_num)

        self.max = np.max(max_num)
        self.min = np.min(min_num)
        self.max_gcm3 = max_num
        self.min_gcm3 = min_num

        # saving results of lower differences
        self.all_diffs_GCM = all_euclidean_distances

        # normalizing
        normalize_values_for_color = []
        for i in all_euclidean_distances:
            colors = (i - np.min(min_num)) / (np.max(max_num) - np.min(min_num))
            normalize_values_for_color.append(colors)

        # plotting distance plots for GCM ----------------------------------------------------------------------------
        x = []
        y = []
        for j in base:
            x.append(j[0])
            y.append(j[1])

        fig, ax = plt.subplots(1, 3)
        fig.suptitle('GCM Results')
        ax[0].scatter(x, y, c=normalize_values_for_color[0])
        ax[0].invert_yaxis()
        ax[1].scatter(x, y, c=normalize_values_for_color[1])
        ax[1].invert_yaxis()
        ax[2].scatter(x, y, c=normalize_values_for_color[2])
        ax[2].invert_yaxis()
        ax[0].title.set_text("Neutral/Smile")
        ax[1].title.set_text("Neutral/Wow")
        ax[2].title.set_text("Neutral/Frown")
        plt.show()
        # ---------------------------------------for color heatmap legend
        # uniform_data = np.random.rand(10, 12)
        # ax = sns.heatmap(uniform_data, linewidth=0.5, cmap="viridis")
        # plt.show()
        # ------------------------------------------------------------------------------------------------------------

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
                    # print("STARTING IMG1/2")
                    # print("UPPER", np.mean(np.array(j[::2])), np.mean(np.array(j[1::2])))
                    # print("LOWER", np.mean(lower_dict['dr']), np.mean(lower_dict['dl']))
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
        upper_dl_avg = []
        upper_dr_avg = []
        lower_dl_avg = []
        lower_dr_avg = []
        for i in range(0, len(upper_dict['dr'])):   # getting length of images to do GCM
            # print("UPPER DR", len(upper_dict['dr'][i]), upper_dict['dr'][i])
            # print("UPPER DL", len(upper_dict['dl'][i]), upper_dict['dl'][i])
            r_upper.append(abs(1 - upper_dict['dl'][i]/upper_dict['dr'][i]))
            r_lower.append(abs(1 - lower_dict['dl'][i]/lower_dict['dr'][i]))
            upper_dl_avg.append(np.mean(upper_dict['dl'][i]))
            upper_dr_avg.append(np.mean(upper_dict['dr'][i]))
            lower_dl_avg.append(np.mean(lower_dict['dl'][i]))
            lower_dr_avg.append(np.mean(lower_dict['dr'][i]))
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

        print("UPPER AVG. DR:", upper_dr_avg)
        print("UPPER AVG. DL:", upper_dl_avg)
        print("LOWER AVG. DR:", lower_dr_avg)
        print("LOWER AVG. DL:", lower_dl_avg)

        print("UPPER AVG R:", avg_r_upper)
        print("LOWER AVG R:", avg_r_lower)
        print("Weighted Average R:", avg_r_all)
        return avg_r_all, avg_r_upper, avg_r_lower, upper_dr_avg, upper_dl_avg, lower_dr_avg, lower_dl_avg

    def GCM2(self):
        # resetting the values from GCM1

        print('--------------------------------------------STARTING GCM2----------------------------------------------')
        # Geometric_Computation.total_diffs(self)
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
            r_upper.append(upper_dict['dl'][i] - upper_dict['dr'][i])
            r_lower.append(lower_dict['dl'][i] - lower_dict['dr'][i])
        for i in range(0, len(all_dict['dr'])):
            r_all.append(all_dict['dl'][i] - all_dict['dr'][i])

        avg_r_upper = []
        avg_r_lower = []
        avg_r_all = []
        for i in range(0, len(r_upper)):
            avg_r_upper.append(np.mean(r_upper[i]))
            avg_r_lower.append(np.mean(r_lower[i]))
        for i in range(0, len(r_all)):
            avg_r_all.append(np.mean(r_all[i]))

        print("UPPER:", avg_r_upper)
        print("LOWER:", avg_r_lower)
        print("Weighted Average:", avg_r_all)
        return avg_r_all, avg_r_upper, avg_r_lower

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
        normalize_values_for_color = []
        for i in distances:
            colors = (i - np.min(i)) / (np.max(i) - np.min(i))
            normalize_values_for_color.append(colors)

            high = np.mean(i[0:upper_num - 1])
            low = np.mean(i[upper_num:])

            upper_lower.append([high, low])
            average_all.append(np.mean(i))

        # plotting distance plots for ICP ----------------------------------------------------------------------------
        # self.normalize_values_for_color = normalize_values_for_color
        # for idx, i in enumerate(self.norm_array_dicts[0:4]):
        #     x = []
        #     y = []
        #     for j in i:
        #         x.append(j[0])
        #         y.append(j[1])
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111)
        #     ax.scatter(x, y, c=self.normalize_values_for_color[idx])
        #     plt.gca().invert_yaxis()
        #     plt.show()
        # -------------------------------------------------------------------------------------------------------------
        self.original_mirrored_distances_alternate = distances
        self.all_average_alternate = average_all
        self.upper_lower_splits_alternate = upper_lower

        return distances, average_all, upper_lower

    def GCM3(self):
        # resetting the values from GCM1
        # GCM3 handles like GCM1 except there is a normalization step that accounts for how much movement there was
        # across the expressions. This is the 'ratio'.

        print('--------------------------------------------STARTING GCM3----------------------------------------------')
        # Geometric_Computation.total_diffs(self)
        upper = self.upper_diffs_GCM
        lower = self.lower_diffs_GCM
        both = self.all_diffs_GCM
        ratios = np.subtract(self.max_gcm3, self.min_gcm3)
        print("RATIOS", ratios)

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
            r_upper.append(abs(1 - upper_dict['dl'][i] / upper_dict['dr'][i]))
            r_lower.append(abs(1 - lower_dict['dl'][i] / lower_dict['dr'][i]))

        for i in range(0, len(all_dict['dr'])):
            r_all.append(abs(1 - all_dict['dl'][i] / all_dict['dr'][i]))

        avg_r_upper = []
        avg_r_lower = []
        avg_r_all = []
        for i in range(0, len(r_upper)):
            avg_r_upper.append(np.mean(r_upper[i]))
            avg_r_lower.append(np.mean(r_lower[i]))
        for i in range(0, len(r_all)):
            avg_r_all.append(np.mean(r_all[i]))

        new_avg_r_upper = np.multiply(avg_r_upper, ratios)
        new_avg_r_lower = np.multiply(avg_r_lower, ratios)
        new_avg_r_all = np.multiply(avg_r_all, ratios)

        print("UPPER", new_avg_r_upper)
        print("LOWER", new_avg_r_lower)
        print("Weighted Average", new_avg_r_all)

        return new_avg_r_all, new_avg_r_upper, new_avg_r_lower, ratios

    def GCM3_2(self):
        # resetting the values from GCM1
        # GCM3 handles like GCM1 except there is a normalization step that accounts for how much movement there was
        # across the expressions. This is the 'ratio'.

        print('--------------------------------------------STARTING GCM3.2--------------------------------------------')
        # Geometric_Computation.total_diffs(self)
        upper = self.upper_diffs_GCM
        lower = self.lower_diffs_GCM
        both = self.all_diffs_GCM
        ratios = np.subtract(self.max_gcm3, self.min_gcm3)

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
            r_upper.append(upper_dict['dl'][i] - upper_dict['dr'][i])
            r_lower.append(lower_dict['dl'][i] - lower_dict['dr'][i])
        for i in range(0, len(all_dict['dr'])):
            r_all.append(all_dict['dl'][i] - all_dict['dr'][i])

        avg_r_upper = []
        avg_r_lower = []
        avg_r_all = []
        for i in range(0, len(r_upper)):
            avg_r_upper.append(np.mean(r_upper[i]))
            avg_r_lower.append(np.mean(r_lower[i]))
        for i in range(0, len(r_all)):
            avg_r_all.append(np.mean(r_all[i]))

        new_avg_r_upper = np.multiply(avg_r_upper, ratios)
        new_avg_r_lower = np.multiply(avg_r_lower, ratios)
        new_avg_r_all = np.multiply(avg_r_all, ratios)

        print("UPPER", new_avg_r_upper)
        print("LOWER", new_avg_r_lower)
        print("Weighted Average", new_avg_r_all)

        return new_avg_r_all, new_avg_r_upper, new_avg_r_lower
