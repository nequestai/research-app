import matplotlib.pyplot as plt
from Code.face_mesh_mediapipe import MediaPipe_Method
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

        self.pop_refs()

    def pop_refs(self):
        ref = []
        mid = []
        for i in self.dicts:
            mid_temp = []
            ref.append([i.pop(self.refs[0]), i.pop(self.refs[1])])
            for mid_point in self.center:
                mid_temp.append(i.pop(mid_point))
            mid.append(mid_temp)
        self.refs = ref
        self.mid = mid

        self.show_dicts()
        self.normalize_dicts()

    def show_dicts(self):
        for idx, dic in enumerate(self.dicts):
            print("IMG", idx, "|", len(dic.items()), "Landmarks |", dic.items())

    def factor_dicts(self):
        # getting refs distance
        difs1 = np.subtract(np.array(self.refs[0][0]), np.array(self.refs[0][1]))
        difs2 = np.subtract(np.array(self.refs[1][0]), np.array(self.refs[1][1]))
        factor1 = math.sqrt(difs1[0] ** 2 + difs1[1] ** 2)
        factor2 = math.sqrt(difs2[0] ** 2 + difs2[1] ** 2)
        # normalizing with respect to refs
        numpy_array_dict1 = np.array(list(self.dicts[0].values()))
        numpy_array_dict2 = np.array(list(self.dicts[1].values()))
        self.norm_array_dict1 = numpy_array_dict1 / factor1
        self.norm_array_dict2 = numpy_array_dict2 / factor2
        print("Dictionaries are now normalized with respect to:")
        print("1:", self.refs[0], "Factor:", factor1, "\n2:", self.refs[1], "Factor:", factor2)

    def new_factor_dicts(self):
        factors = []
        for ref in self.refs:
            dif = np.subtract(np.array(ref[0]), np.array(ref[1]))
            factors.append(math.sqrt(dif[0]**2 + dif[1]**2))
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
        return out

    def normalize_dicts(self):
        # get normalizing factors from existing images
        factors = Geometric_Computation.new_factor_dicts(self)
        # getting sagittal line
        sagittal = Geometric_Computation.sagittalize(self)

        for idx, mp_dict in enumerate(self.dicts):
            np_array_dict = np.array(list(mp_dict.values()))
            self.norm_array_dicts.append(np_array_dict / factors[idx])
            self.mid[idx] = np.array(sagittal[idx]) / factors[idx]

        print("Normalization of", len(self.norm_array_dicts), "dictionaries is complete.")
        print("Sagittal line reference computed")

    def mid_norm_plot(self, plot=True):
        for idx, norm_dict in enumerate(self.norm_array_dicts):
            self.norm_array_dicts[idx] = self.norm_array_dicts[idx] - self.mid[idx]
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
            self.upper_splits.append(i[0:upper_num-1])
            self.lower_splits.append(i[upper_num:upper_num+lower_num])
        print(len(self.upper_splits)*2, "lower & upper splits successful")

    def upper_diffs(self):
        base = self.upper_splits.pop(0)
        upper_euclidean_distances = []
        for idx, i in enumerate(self.upper_splits):
            upper_euclidean_distances.append(np.linalg.norm(base - i[idx], axis=1))
        return upper_euclidean_distances

    def lower_diffs(self):
        base = self.lower_splits.pop(0)
        lower_euclidean_distances = []
        for idx, i in enumerate(self.lower_splits):
            lower_euclidean_distances.append(np.linalg.norm(base - i[idx], axis=1))
        return lower_euclidean_distances

    def left_right_diffs(self):
        Geometric_Computation.regional_split(self)
        upper_diffs = Geometric_Computation.upper_diffs(self)
        lower_diffs = Geometric_Computation.lower_diffs(self)


    @staticmethod
    def compute_icp(reference_points, points):
        transformation_history, aligned_points = icp(reference_points,
                                                     points,
                                                     verbose=True)
        # show results
        plt.plot(reference_points[:, 0], reference_points[:, 1], 'rx', label='original points')
        plt.plot(points[:, 0], points[:, 1], 'b1', label='mirrored points')
        plt.plot(aligned_points[:, 0], aligned_points[:, 1], 'g+', label='aligned points')
        plt.legend()
        plt.gca().invert_yaxis()
        plt.show()
        print("ICP Complete")
        return reference_points, aligned_points

    def get_icp_distances(self, points):

        reference_points = self.norm_array_dicts

        distances = []
        for idx, array in enumerate(reference_points):
            out, aligned = Geometric_Computation.compute_icp(array, points[idx])

        # reference_points, aligned_points = Geometric_Computation.compute_icp(self.norm_array_dicts[0],
        #                                                                      self.norm_array_dicts[1])
        # Using Euclidean distance
        # dist = np.linalg.norm(reference_points-aligned_points, axis=1)
        # print("Euclidean Distance:", dist)

    def all_diffs(self):
        prim = np.subtract(self.norm_array_dict2, self.norm_array_dict1)**2
        sec = []
        for i in range(0, len(prim)):
            sec.append(math.sqrt(prim[i][0] + prim[i][1]))
        dr = np.array(sec[::2])
        dl = np.array(sec[1::2])
        r = abs(1 - dl / dr)     # len(r) = 68
        u_r = np.sum(r) / len(prim)

        self.results = [dr, dl, r, u_r] # this is not averaged
        self.results = [np.mean(self.results[0]), np.mean(self.results[1]), np.mean(self.results[2]), self.results[3]]
        return print("Geometric Difference Complete")

    def show_results(self):
        print("Geometric Computation for landmarks:", self.all_index,
              "\nReference Landmarks:", self.refs,
              "\nAvg. Left Alteration:", np.mean(self.results[1]),
              "\nRight Alteration:", np.mean(self.results[0]),
              "\nR-Value:", np.mean(self.results[2]),
              "\nu_r Value:", self.results[3])
