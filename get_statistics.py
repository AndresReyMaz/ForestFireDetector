import cv2
import numpy as np
from os import listdir
import scipy.stats

def get_statistics():
    # Extracts image features and prints them to stats.txt.
    # The order (for each image) is the following:
    # 1 - Mean of R
    # 2 - Mean of G
    # 3 - Mean of B
    # 4 - Standard deviation of R
    # 5 - Standard deviation of G
    # 6 - Standard deviation of B
    # 7 - Median of R
    # 8 - Median of G
    # 9 - Median of B
    # 10 - Entropy of R
    # 11 - Entropy of G
    # 12 - Entropy of B
    # 13 - Energy of R
    # 14 - Energy of G
    # 15 - Energy of B
    # 16 - Kurtosis of R
    # 17 - Kurtosis of G
    # 18 - Kurtosis of B
    # 19 - Mean of L
    # 20 - Mean of a
    # 21 - Mean of b
    # 22 - Standard deviation of L
    # 23 - Standard deviation of a
    # 24 - Standard deviation of b
    # 25 - Median of L
    # 26 - Median of a
    # 27 - Median of b
    # 28 - Correlation factor between a and b
    
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
        return sorted([f for f in listdir('./kernels')])

    def get_correlation(mean_1, mean_2, arr_1, arr_2, std_1, std_2):
        # Calculates the factor of correlation between two random variables.
        def get_covariance():
            n = len(arr_1)
            numerator = 0
            for i in range(n):
                numerator += (arr_1[i] - mean_1) * (arr_2[i] - mean_2)
            return numerator / (n - 1)

        return get_covariance() / (std_1 * std_2)

    stats_file = open("stats.txt", "w")

    image_names = get_image_names()
    for image_name in image_names:
        measures = []
        image = cv2.imread('kernels/' + image_name)

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

        stats_file.write(",".join(map(str, measures)) + "\n")

get_statistics()