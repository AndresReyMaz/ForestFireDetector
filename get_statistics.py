import argparse
import cv2
import numpy as np
from os import listdir
import scipy.stats

def get_statistics(image_folder_path, output_file):
    # Extracts image features and prints them to output_file.
    
    def get_probability_distribution(matrix, min_elem=0, max_elem=255):
        counts = dict()
        for sublist in matrix:
            for elem in sublist:
                if elem in counts:
                    counts[elem] += 1
                else:
                    counts[elem] = 1
        p_dist = []
        n_elements = len(matrix) * len(matrix[0])
        for i in range(min_elem, max_elem + 1):
            if i in counts:
                p_dist.append(counts[i] / n_elements)
            else:
                p_dist.append(0)
        return p_dist

    def get_energy(p_dist):
        # Calculates the energy given a probability distribution.
        return sum(list(map(lambda x: x**2, p_dist)))


    def get_image_names():
        # Gets all of the filenames for images in the image path.
        return [str(f) + '.jpg' for f in range(1,4001)]

    def get_correlation(mean_1, mean_2, arr_1, arr_2, std_1, std_2):
        # Calculates the factor of correlation between two random variables.
        def get_covariance():
            n = len(arr_1)
            numerator = 0
            for i in range(n):
                numerator += (arr_1[i] - mean_1) * (arr_2[i] - mean_2)
            return numerator / n
        if std_1 == 0 or std_2 == 0:
            return 0
        return get_covariance() / (std_1 * std_2)

    def get_inverse_difference(arr_1, arr_2):
        # Aka "homogeneidad".
        ans = 0
        n = len(arr_1)
        for i in range(n):
            ans += 1 / ((1 + (arr_1[i] - arr_2[i]) ** 2) * n)
        return ans

    def get_max(matrix):
        # Returns the maximimum element in the matrix.
        ans = -1000000000
        for sublist in matrix:
            for elem in sublist:
                ans = max(ans, elem)
        return ans

    def get_min(matrix):
        # Returns the minimum element in the matrix.
        ans = 1000000000
        for sublist in matrix:
            for elem in sublist:
                ans = min(ans, elem)
        return ans

    stats_file = open(output_file, "w")

    image_names = get_image_names()
    for image_name in image_names:
        measures = []
        image = cv2.imread(image_folder_path + '/' + image_name)
        print("Processing", image_name)
        # Get statistics for BGR color space.
        (B, G, R) = cv2.split(image)
        measures.append(np.mean(R))
        measures.append(np.mean(G))
        measures.append(np.mean(B))
        measures.append(np.std(R))
        measures.append(np.std(G))
        measures.append(np.std(B))
        measures.append(np.median(R))
        measures.append(np.median(G))
        measures.append(np.median(B))
        measures.append(get_max(R))
        measures.append(get_max(G))
        measures.append(get_max(B))
        measures.append(get_min(R))
        measures.append(get_min(G))
        measures.append(get_min(B))
        p_dist_R = get_probability_distribution(R)
        p_dist_G = get_probability_distribution(G)
        p_dist_B = get_probability_distribution(B)
        measures.append(scipy.stats.entropy(p_dist_R))
        measures.append(scipy.stats.entropy(p_dist_G))
        measures.append(scipy.stats.entropy(p_dist_B))
        measures.append(get_energy(p_dist_R))
        measures.append(get_energy(p_dist_G))
        measures.append(get_energy(p_dist_B))
        measures.append(scipy.stats.kurtosis(np.ndarray.flatten(R)))
        measures.append(scipy.stats.kurtosis(np.ndarray.flatten(G)))
        measures.append(scipy.stats.kurtosis(np.ndarray.flatten(B)))

        # Get statistics for L*a*b color space.
        Lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        (L, a, b) = cv2.split(Lab)
        measures.append(np.mean(L))
        mean_a = np.mean(a) 
        measures.append(mean_a)
        mean_b = np.mean(b)
        measures.append(mean_b)
        measures.append(np.std(L))
        std_a = np.std(a)
        measures.append(std_a)
        std_b = np.std(b)
        measures.append(std_b)
        measures.append(np.median(L))
        measures.append(np.median(a))
        measures.append(np.median(b))
        measures.append(get_correlation(mean_a, mean_b, np.ndarray.flatten(a), np.ndarray.flatten(b), std_a, std_b))

        # Get statistics for grayscale.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        measures.append(np.mean(gray))

        # Get statistics for HSV.
        HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        (H, S, V) = cv2.split(HSV)
        measures.append(np.mean(H))
        measures.append(np.mean(S))
        measures.append(np.mean(V))
        measures.append(np.std(H))
        measures.append(np.std(S))
        measures.append(np.std(V))
        measures.append(get_inverse_difference(np.ndarray.flatten(L), np.ndarray.flatten(R)))

        stats_file.write(",".join(map(str, measures)) + "\n")


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--imagefolder", required = True,help = "Path to the image folder")
ap.add_argument("-o", "--output", required = True, help = "Output file")
args = vars(ap.parse_args())

get_statistics(args['imagefolder'], args['output'])
# weka.classifiers.functions.MultilayerPerceptron -L 0.099 -M 0.70068 -N 500 -V 0 -S 0 -E 20 -H 15