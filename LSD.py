import numpy as np                     # Numerical algorithms on arrays
import cv2                             # OpenCV
from pylsd.lsd import lsd              # LSD.py python binding
import math                            # Convert radian to degree
from sklearn.cluster import AgglomerativeClustering  # Hierarchical clustering
from scipy.spatial import distance  # Euclidean distance computation
import edge_detector as ed
import segment_detector as sd


def lsd_alg(color_image, line_width=0, fuse=False, dTheta=2 / 360 * np.pi * 2, dRho=2, maxL=4):
    """
    LSD algoritm for line segment detection
    :param color_image: [np.array] The input image in BGR mode
            line_width: override line width during drawing result
           fuse:   [bool] Whether to fuse together the segments detected
           dTheta: [float] The max difference in theta between two segments to be fused together
           dRho:   [float] The max difference in rho between two segments to be fused together
           maxL:   [int] The maximal number of lines fused together. Set to 0 for no limit.
    :return:    lines: [np.array] (nb_lines, 4) each element of the array corresponds to [pt1_x, pt1_y, pt2_x, pt2_y] in DOUBLES
		        result_lines: [np.array] BGR images with red lines representing LSD result
                result_points: [np.array] BGR images with red dots representing LSD result
    """

    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    result_lines = np.zeros(gray_image.shape + (3,))  # Black RGB image with same height/width than gray-image
    result_points = np.zeros(gray_image.shape + (3,))
    lines = lsd(gray_image)  # python script calling the C++ so library
    
    # Fuse lines if asked
    if fuse:
        lines = lines.reshape((lines.shape[0],1,lines.shape[1]))
        lines = sd.fuseCloseSegment(lines, dTheta, dRho, maxL)
        lines = lines.reshape((lines.shape[0],lines.shape[2]))
    

    for i in range(lines.shape[0]):
        pt1 = (int(lines[i, 0]), int(lines[i, 1]))
        pt2 = (int(lines[i, 2]), int(lines[i, 3]))
        if line_width == 0:
            width = lines[i, 4]
        else:
            width = line_width * 2
        cv2.line(result_lines, pt1, pt2, (255, 255, 255), int(np.ceil(width / 2)))

        if 0 <= pt1[0] < gray_image.shape[1] and 0 <= pt1[1] < gray_image.shape[0]:   # Some coordinates returned can be out of bounds
            result_points[pt1[1]][pt1[0]][2] = 255  # Add a red pixel for each end-point of a line
        if 0 <= pt2[0] < gray_image.shape[1] and 0 <= pt2[1] < gray_image.shape[0]:
            result_points[pt2[1]][pt2[0]][2] = 255  # Add a red pixel for each end-point of a line

    return lines[:, :4], result_lines, result_points  # Lines over a Black background


def lsd_getAxis(color_image):
    """
    This function proceed to a hierachical clustering based on the LSD algo results
    :param color_image: [np.array] The input image in BGR mode
    :return: final_result: The average angle of each cluster
             clusters_nb_elem: The number of elements for each cluster
    """

    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    lines = lsd(gray_image)  # python script calling the C++ so library
    deltas = list()
    deltas_points = list()  # Keep track of points for duplicated value

    for i in range(lines.shape[0]):
        pt1 = (int(lines[i, 0]), int(lines[i, 1]))
        pt2 = (int(lines[i, 2]), int(lines[i, 3]))

        # Getting deltas for future cosines hierarchical clustering.
        # Add line data several time according to their "weight" (euclidean distance)
        # A additional representation by 5% step of the main diagonal euclidean distance
        # The y axis is invert compare to the cartesian one (invert pt1 and pt2 for y)
        if pt2[0] - pt1[0] >= 0:
            for j in range(int(np.ceil(15 * distance.euclidean(pt1, pt2)/distance.euclidean((0, 0), gray_image.shape)))):
                deltas.append([pt2[0] -  pt1[0], pt1[1] - pt2[1]])
                deltas_points.append((pt1, pt2))
        else:
            for j in range(int(np.ceil(15 * distance.euclidean(pt1, pt2)/distance.euclidean((0, 0), gray_image.shape)))):
                deltas.append([-(pt2[0] - pt1[0]), -(pt1[1] - pt2[1])])  # Corresponding orientation in the first and fourth quadrant
                deltas_points.append((pt1, pt2))

    deltas = np.array(deltas)
    # 4 clusters because of the discountinuity for +90° and -90°
    # cosine affinity compute distance with the angle between the input vector
    clustering = AgglomerativeClustering(n_clusters=4, affinity='cosine', linkage='average')
    clustering.fit(deltas)
    clusters_result = np.zeros((clustering.n_clusters, 2))
    clusters_nb_elem = np.zeros(clustering.n_clusters)

    for i in range(len(clustering.labels_)):

        clusters_result[clustering.labels_[i]] = clusters_result[clustering.labels_[i]] + deltas[i]
        clusters_nb_elem[clustering.labels_[i]] = clusters_nb_elem[clustering.labels_[i]] + 1

    final_result = list()
    for j in range(len(clusters_nb_elem)):
        # arctan( tan = delta y / delta x )
        final_result.append(math.degrees(np.arctan(clusters_result[j][1] / clusters_result[j][0])))

    return final_result, clusters_nb_elem  # Lines over a Black background
    
