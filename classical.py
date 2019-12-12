import cv2
import numpy as np
        
"""
    Yuen, H. K. and Princen, J. and Illingworth, J. and Kittler, J., 
    Comparative study of Hough transform methods for circle finding. 
    Image Vision Comput. 8 1, pp 71–77 (1990)
"""
def houghCircles(image, minRadius, maxRadius, dp=1, minDist=100, threshold=20):
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp, minDist, 50,
                               threshold, minRadius, maxRadius)
    
    ellipses = []
    
    for circle in circles[0, :]:
        ellipse = [circle[0], circle[1], 1/circle[2], 0, 1/circle[2]]
        ellipses.append(ellipse)
    return ellipses

def distance(pt1, pt2):
    y1, x1 = pt1
    y2, x2 = pt2
    sqdist = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
    return (sqdist)**(1/2), sqdist

# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1.8792&rep=rep1&type=pdf
def youghongQiangEllipse(image, minDist, maxDist, minLength, maxLength, bags, threshold, ratio=0.15):
    """ 
    Evaluates the ellipse parameters present in a binary contour image
    by looking for plausible main axes and accumulating secondary axes.
    
    @Args:
        image:     [np.array] binary contour image
        minDist:   [numeric] The minimum length for the main axis
        maxDist:   [numeric] The maximum length for the main axis
        minLength: [numeric] The minimum length for the secondary axis
        maxLength: [numeric] The maximum length for the secondary axis
        bags:      [int] The number of bags used for the accumulator. The 
                   accumulator resolution is (maxLength - minLength) / bags.
        threshold: [int] The number of hits in an accumulator bag needed for this
                   bag to be validated as an actual ellipse parameter.
        ratio:     [float] The percentage of pixels of the contour to be randomly 
                   taken. None to take everything.
    @Return:
        a list of ellipses in the format 
        [[(centerX, centerY), (mainHalfLength, secondaryHalfLength), angle)] ...]
    """
    pixels = np.random.permutation(np.argwhere(image==255))
    
    if ratio is not None:
        n_pixels = int(ratio * len(pixels))
        pixels = pixels[:n_pixels]   
    accLength = maxLength - minLength
    
    ellipses = []
    
    for p1 in pixels:
        for p2 in pixels:
            
            accumulator = np.zeros((bags,))
            not_accumulated_pixels = []
            
            dist, distSq = distance(p1, p2)
            if dist > minDist and dist < maxDist:
                p0    = (p1 + p2) / 2
                a     = dist / 2
                aSq   = distSq / 4

                for p3 in pixels:
                    d, dSq = distance(p0, p3)
                    if d > minDist and d < a:
                        f, fSq = distance(p3, p2)
                        cosTau = (aSq + dSq - fSq) / (2 * a * d)
                        cosTauSq = cosTau * cosTau
                        if aSq - dSq * cosTauSq != 0:
                            bSq = ((aSq * dSq) * (1 - cosTauSq)) / (aSq - dSq * cosTauSq)
                            bSq = 0 if bSq < 0 else bSq  
                            b   = bSq**(1/2)
                          
                            if b > minLength and b < maxLength:
                                bag_id = int((b - minLength) * (bags / accLength))
                                accumulator[bag_id] += 1
                            else:
                                not_accumulated_pixels.append(p3)
                    else:
                        not_accumulated_pixels.append(p3)
                            
                best = np.argmax(accumulator)
                if accumulator[best] > threshold:
                    
                    b = minLength + best * (accLength / bags)   
                    y1, x1 = p1
                    y2, x2 = p2  
                    y0, x0 = p0
                    alpha = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    ellipse = [(int(x0), int(y0)), (int(a), int(b)), alpha] 
                    ellipses.append(ellipse)
                    
                    pixels = np.array(not_accumulated_pixels)
    return ellipses
