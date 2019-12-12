import cv2
import numpy as np
from imgTools import display, multiDisplay
#import segment_detector as sd
        
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
    
if __name__ == "__main__":
    #image = cv2.imread("images_database/eyes/elps_eye01_2014-11-26_08-49-31-060.png", cv2.IMREAD_GRAYSCALE)
    image = cv2.imread("images_database/eyes/elps_eye03_2014-12-09_02-42-00-002.png", cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image,(360, 200)) # Images resized to 360p
    _, imageBinary = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
    imageBinary = cv2.Canny(image, 20, 100)
    #(image, n_pixels, minDist, maxDist, minLength, maxLength, bags, threshold)
    ellipses1 = youghongQiangEllipse(imageBinary, 15, 50, 5, 25, 40, 10, 0.5)
    
    print(len(ellipses1))
     
    imageBGR = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for ell in ellipses1:
        cv2.ellipse(imageBGR, ell[0], ell[1], ell[2], 0, 360, (0, 0, 255))
    

    display("", imageBinary)
    multiDisplay(["", ""], [image, imageBGR], 2)
    #cv2.imshow("test", imageBGR)
    #cv2.waitKey(50000)
