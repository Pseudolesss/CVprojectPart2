# =====================================================================================================================
# Segment and endpoints detector for the first part
# =====================================================================================================================

import cv2
import numpy as np
import random
import edge_detector as ed
import LSD


def segHough(input_img, fctEdges, rho=1, theta=np.pi / 180, thresh=50,
             minLineLen=5, maxLineGap=0, kSize=2,
             fuse=False, dTheta=2 / 360 * np.pi * 2, dRho=2, maxL=3,
             lineWidth=1, dilate=True):
    """
    Apply the segment detection by preprocessing the image with the edge detection and using the Probabilistic Hough 
    Transform.

    @Args:
        input_img:        [np.array] The image.
        fctEdges:    [python function] Function taking the img as argument and returning the edge detection of the image.
                    The edges are of value 255 and the rest is at 0.
        rho:        [double] resolution of the image
        theta:         [double] The resolution of the parameter in radians. We use 1 degree
        thresh:  [int] The minimum number of intersections to “detect” a line
        minLineLen: [double] The minimum number of points that can form a line. Lines with less than this number of
                    points are disregarded.
        maxLineGap: [double] The maximum gap between two points to be considered in the same line.
        kSize:        [int] Size of kernel for dilation
        fuse:        [bool] Fuse toghether close segments.
        dTheta:     [float] The max difference in theta between two segments to be fused together
        dRho:       [float] The max difference in rho between two segments to be fused together
        maxL:       [int] The maximal number of lines fused together. Set to 0 for no limit.
        dilate:        [bool] Whether to dilate the edge after detecting them or not.

    @Return:
        img_edges       [np.array] the image with the edges
        lines_p:        [numpy array of shape (num seg x 1 x 4)] Array containing the coordinates of the first and second 
                        endpoint of segment of line.
        img_edges_segment:    [np.array] Image containing the edges and segments
        img_segment:    [np.array] Image containing the segments
    """
    # Detect the edges
    img_edges = fctEdges(input_img)

    # Dilate edges
    if dilate:
        kernel = np.ones((kSize, kSize), np.uint8)
        img_edges = cv2.dilate(img_edges, kernel,
                               borderType=cv2.BORDER_CONSTANT,
                               iterations=1)

    # Detect segments of lines
    lines_p, img_edges_segment, img_segment = hough(img_edges, rho, theta,
                                                    thresh, minLineLen,
                                                    maxLineGap, fuse,
                                                    dTheta, dRho, maxL,
                                                    lineWidth)

    return img_edges, lines_p, img_edges_segment, img_segment


def hough(input_img, rho=1, theta=np.pi / 180, thresh=50, minLineLen=5,
          maxLineGap=0,
          fuse=False, dTheta=2 / 360 * np.pi * 2,
          dRho=2, maxL=3, lineWidth=1):
    """
    Apply the probabilistic Hough Transform on the image.

    @Args:
        input_img:        [np.array] The image with detection of edges.
        rho:        [double] resolution of the image
        theta:         [double] The resolution of the parameter in radians. We use 1 degree
        thresh:      [int] The minimum number of intersections to “detect” a line
        minLineLen: [double] The minimum number of points that can form a line. Lines with less than this number of
                    points are disregarded.
        maxLineGap: [double] The maximum gap between two points to be considered in the same line.
        fuse:        [bool] Fuse toghether close segments.
        dTheta:     [float] The max difference in theta between two segments to be fused together
        dRho:       [float] The max difference in rho between two segments to be fused together
        maxL:       [int] The maximal number of lines fused together. Set to 0 for no limit.
        lineWidth:  [int] The width of segments drawn.

    @Return:
        lines_p:        [numpy array of shape (num seg x 1 x 4)] Array containing the coordinates of the first and second 
                        endpoint of segment of line.
        img_edges_segment:    [np.array] the image of the segment detected with the edges detected previously
        img_segment:    [np.array] the image of the segments detected only
    """
    # Copy edges to the images that will display the results in BGR
    img_edges_segment = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
    img_segment = input_img * 0

    # Detect segment of lines
    lines_p = cv2.HoughLinesP(input_img, rho=rho, theta=theta,
                              threshold=thresh,
                              minLineLength=minLineLen,
                              maxLineGap=maxLineGap)

    # Fuse lines if asked
    if fuse:
        lines_p = fuseCloseSegment(lines_p, dTheta, dRho)

    # Add segment detected to images
    if lines_p is not None:
        for i in range(0, len(lines_p)):
            line = lines_p[i][0]
            cv2.line(img_edges_segment, (line[0], line[1]), (line[2], line[3]),
                     (0, 0, 255), lineWidth)
            cv2.line(img_segment, (line[0], line[1]), (line[2], line[3]),
                     255, lineWidth)
    return lines_p, img_edges_segment, img_segment


def toHoughSpaceVariant(AB):
    """
    Given the list of the two end points of segments, return the values of the segments in a variant of the Hough space.
    @Args:
        AB:        [numpy array of shape (num seg x 1 x 4)] Array containing the coordinates of the first and second
                endpoint of segment of line. (Considering the origin in top left and values in order [v1, h1, v2, h2].)
    @Return:
        A list of lists containing :
        theta:    [float] inclination of the slope of the segment in radians
        rho:    [float] shortest distance between the segment (extended to infinity) and the origin.
        p:        [float] distance from C to the endpoint with the lowest horizontal value. C being the intersection
                between the segment extended and the perpendicular to the segment going to rho
        d:        [float] distance from A to B
    """
    retList = []

    for i in range(AB.shape[0]):
        endpts = AB[i]
        av = endpts[0][0]  # vertical coord. of A
        ah = endpts[0][1]  # horizontal coord. of A
        bv = endpts[0][2]  # vertical coord. of B
        bh = endpts[0][3]  # horizontal coord. of B

        d = np.linalg.norm(np.array([bv - av, bh - ah]))
        if bh - ah == 0:
            theta = np.pi / 2
        else:
            theta = np.arctan(
                abs(bv - av) / abs(bh - ah))  # theta in [0, pi/2[
            if (ah > bh and av > bv) or (
                    ah < bh and av < bv):  # decreasing slope => theta in ]pi/2, pi[
                theta = np.pi - theta

        rho = np.linalg.norm(
            np.cross(np.array([bv - av, bh - ah]), np.array([av, ah]))) / d
        if av != bv and ah - av * (ah - bh) / (av - bv) < 0:
            rho = -rho  # rho < 0 if ch < 0
        cv = rho * np.cos(theta)
        ch = rho * np.sin(theta)

        if ah < bh or (ah == bh and av > bv):
            p = np.sign(-5 + 6 * np.sign(ah - ch)) * np.linalg.norm(
                np.array([av - cv, ah - ch]))  # p < 0 if ah <= ch
        else:
            p = np.sign(-5 + 6 * np.sign(bh - ch)) * np.linalg.norm(
                np.array([bv - cv, bh - ch]))  # p < 0 if bh <= ch

        retList.append([theta, rho, p, d])

    return retList


def fromHoughSpaceVariant(abHS):
    """
    From Hough space variant to the segment endpoints.
    @Args:
        A list of lists containing in this order:
        theta:    [float] inclination of the slope of the segment in radians
        rho:    [float] shortest distance between the segment (extended to infinity) and the origin.
        p:        [float] distance from C to the endpoint with the lowest horizontal value. C being the intersection
                between the segment extended and the perpendicular to the segment going to rho
        d:        [float] distance from A to B
    @Return:
        AB:        [numpy array of shape (num seg x 1 x 4)] Array containing the coordinates of the first and second
                endpoint of segment of line. (Considering the origin in top left and values in order [v1, h1, v2, h2].)
    """
    retList = np.zeros((len(abHS), 1, 4), dtype=int)

    for i in range(len(abHS)):
        pt = abHS[i]
        theta = pt[0]
        rho = pt[1]
        p = pt[2]
        d = pt[3]
        sin = np.sin(theta)
        cos = np.cos(theta)

        # finding C,
        # C being the intersection between the segment extended and the perpendicular to the segment going to rho
        cv = cos * rho  # vertical coord of C
        ch = sin * rho  # horizontal coord. of C

        # Finding the endpoints A and B
        av = int(
            round(cv - np.sign(1 + 2 * np.sign(np.pi / 2 - theta)) * p * sin))
        ah = int(
            round(ch + np.sign(1 + 2 * np.sign(np.pi / 2 - theta)) * p * cos))
        bv = int(round(
            cv - np.sign(1 + 2 * np.sign(np.pi / 2 - theta)) * (p + d) * sin))
        bh = int(round(
            ch + np.sign(1 + 2 * np.sign(np.pi / 2 - theta)) * (p + d) * cos))

        retList[i][0] = np.array([av, ah, bv, bh])

    return retList


def fuseCloseSegment(AB, dTheta=2 / 360 * np.pi * 2, dRho=2, maxL=0):
    """
    Fuse close segments together.
    @Args:
        AB:        [numpy array of shape (num seg x 1 x 4)] Array containing the coordinates of the first and second
                endpoint of segment of line. (Considering the origin in top left and values in order [v1, h1, v2, h2].)
        dTheta: [float] The max difference in theta between two segments to be fused together
        dRho:   [float] The max difference in rho between two segments to be fused together
        maxL:   [int] The maximal number of lines fused together. Set to 0 for no limit.
    @Return:
        The list of segment with close segments fused together.
    """
    abHS = toHoughSpaceVariant(AB)
    i = 0
    length = len(abHS)
    cnt = 1  # Count of max line fused together

    while True:
        if i == len(abHS):
            break
        elif i == length:  # another line has been fused
            cnt += 1
            length = len(abHS)
            i = 0
            if cnt == maxL:
                break

        seg1 = abHS[i]
        removeI = False
        toAdd = []
        toRemove = []
        for j in range(i + 1, len(abHS)):
            seg2 = abHS[j]

            # Check first condition to fuse (position and orientation of lines)
            if abs(seg1[0] - seg2[0]) <= dTheta and abs(
                    seg1[1] - seg2[1]) <= dRho:
                p1 = seg1[2]
                d1 = seg1[3]
                p2 = seg2[2]
                d2 = seg2[3]

                # Check second condition (position of segments on the lines)
                if ((p1 - dRho / 2 <= p2 and p1 + d1 + dRho / 2 > p2) or
                        (p1 <= p2 and p1 + d1 + dRho > p2) or
                        (p2 - dRho / 2 <= p1 and p2 + d2 + dRho / 2 > p1) or
                        (p2 <= p1 and p2 + d2 + dRho > p1)):
                    removeI = True

                    newTheta = (seg1[0] + seg2[0]) / 2
                    newRho = (seg1[1] + seg2[1]) / 2
                    newP = min(p1, p2)
                    newD = max(p1 + d1, p2 + d2) - newP

                    toAdd.append([newTheta, newRho, newP, newD])
                    toRemove.append(seg2)
                    break

        # increment new seg to check
        if removeI:
            abHS.remove(seg1)
            length -= 1
        else:
            i += 1

        # Add new fused segment and remove previous ones
        for seg in toRemove:
            abHS.remove(seg)
            length -= 1
        for seg in toAdd:
            abHS.append(seg)

    return fromHoughSpaceVariant(abHS)


def segmentDetectorFinal(input_img, dataset=None, lineWidth=2):
    """
    The segment detector chosen finally after comparing the different candidates
    :param input_img: [np.array] The input image
    :param dataset:   [str] The dataset from which the image origins from the list ['sudoku'].
    :param lineWidth:  [int] The width of segments drawn.
    :return:    img_edges       [np.array] the image with the edges
                lines_p:        [numpy array of shape (num seg x 1 x 4)] Array containing the coordinates of the first and second
                                endpoint of segment of line.
                img_edges_segment:  [np.array] the image of the segment detected with the edges detected previously
                img_segment:        [np.array] the image of the segments detected only
    """
    if not dataset is None:  # particular dataset used
        if dataset == 'sudoku':
            img_edges = ed.canny_median_blur(input_img, downsize=False)
            lines, img_segment, img_points = LSD.lsd_alg(input_img,
                                                         line_width=lineWidth,
                                                         fuse=True,
                                                         dTheta=1 / 360 * np.pi * 2,
                                                         dRho=8,
                                                         maxL=4)
            lines = lines.reshape((lines.shape[0], 1, lines.shape[1]))

            # Add segment detected to the edges image
            img_edges_segment = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2BGR)
            if lines is not None:
                for i in range(0, len(lines)):
                    line = lines[i][0]
                    cv2.line(img_edges_segment, (line[0], line[1]),
                             (line[2], line[3]), (0, 0, 255), lineWidth)

            return img_edges, lines, img_edges_segment, img_segment

        if dataset == 'pcb':
            img_edges = ed.canny_gaussian_blur_downsize(input_img,
                                                        lo_thresh=150,
                                                        hi_thresh=200,
                                                        sobel_size=3,
                                                        i_gaus_kernel_size=5,
                                                        gauss_center=1)

            lines, img_edges_segment, img_segment = hough(img_edges, 1,
                                                          np.pi / 180,
                                                          thresh=10,
                                                          minLineLen=7,
                                                          maxLineGap=3,
                                                          fuse=True,
                                                          dTheta=3 / 360 * np.pi * 2,
                                                          dRho=3, maxL=3,
                                                          lineWidth=lineWidth)

            return img_edges, lines, img_edges_segment, img_segment

        if dataset == 'soccer':
            img_edges = ed.canny_median_blur(input_img, downsize=False)
            hsv = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
            low = np.array([30, 0, 150])
            upp = np.array([90, 70, 255])
            mask = cv2.inRange(hsv, low, upp)
            img_mask = cv2.bitwise_and(input_img, input_img, mask=mask)
            ret = cv2.cvtColor(img_mask, cv2.COLOR_HSV2BGR)
            # LSD
            lines, img_segment, img_points = LSD.lsd_alg(ret)
            lines = lines.reshape((lines.shape[0], 1, lines.shape[1]))
            lines = np.around(lines).astype(int)

            # Add segment detected to the edges image
            img_edges_segment = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2BGR)
            if lines is not None:
                for i in range(0, len(lines)):
                    line = lines[i][0]
                    cv2.line(img_edges_segment, (line[0], line[1]),
                             (line[2], line[3]), (0, 0, 255), lineWidth)

            return img_edges, lines, img_edges_segment, img_segment

        if dataset == 'road':                            
            img_edges = ed.edgesDetectionFinal(input_img)
            lines, img_segment, img_points = LSD.lsd_alg(input_img,
                                                         line_width=lineWidth,
                                                         fuse=True,
                                                         dTheta=1 / 360 * np.pi * 2,
                                                         dRho=5,
                                                         maxL=4)
            lines = lines.reshape((lines.shape[0], 1, lines.shape[1]))
            lines = np.around(lines).astype(int)

            # Add segment detected to the edges image
            img_edges_segment = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2BGR)
            if lines is not None:
                for i in range(0, len(lines)):
                    line = lines[i][0]
                    cv2.line(img_edges_segment, (line[0], line[1]),
                             (line[2], line[3]), (0, 0, 255), lineWidth)

            return img_edges, lines, img_edges_segment, img_segment
            
        if dataset == 'building':
            img_edges = ed.edgesDetectionFinal(input_img)
            lines, img_segment, img_points = LSD.lsd_alg(input_img,
                                                         line_width=lineWidth,
                                                         fuse=True,
                                                         dTheta=1 / 360 * np.pi * 2,
                                                         dRho=2,
                                                         maxL=4)
            lines = lines.reshape((lines.shape[0], 1, lines.shape[1]))
            lines = np.around(lines).astype(int)

            # Add segment detected to the edges image
            img_edges_segment = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2BGR)
            if lines is not None:
                for i in range(0, len(lines)):
                    line = lines[i][0]
                    cv2.line(img_edges_segment, (line[0], line[1]),
                             (line[2], line[3]), (0, 0, 255), lineWidth)

            return img_edges, lines, img_edges_segment, img_segment

    return segHough(input_img, ed.edgesDetectionFinal, lineWidth=lineWidth)
