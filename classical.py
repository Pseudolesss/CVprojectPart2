import cv2
import numpy as np

def ellipseDistance(ell1, ell2):
    center1 = ell1[0]
    axes1 = ell1[1]
    alpha1 = ell1[2]
    
    x1, y1 = center1
    d10, d11 = axes1
    
    center2 = ell2[0]
    axes2 = ell2[1]
    alpha2 = ell2[2]
    
    x2, y2 = center2
    d20, d21 = axes2
    
    distSq = (x1 - x2)**2 + (y1 - y2)**2
    
    return distSq**(1/2)

def sumEllipses(ell1, ell2):
    center1 = ell1[0]
    axes1 = ell1[1]
    alpha1 = ell1[2]
    
    x1, y1 = center1
    d10, d11 = axes1
    
    center2 = ell2[0]
    axes2 = ell2[1]
    alpha2 = ell2[2]
    
    x2, y2 = center2
    d20, d21 = axes2
    
    x = x1 + x2
    y = y1 + y2
    
    d0 = d10 + d20
    d1 = d11 + d21
    
    alpha = alpha1
    
    return [(int(x), int(y)), (int(d0), int(d1)), alpha]

def mergeEllipses(ellipses, dist_threshold, qtt_threshold):
    mergedParams = []
    cnt = []
    
    mergedParams.append(ellipses[0])
    cnt.append(1)
    
    for ellipse in ellipses[1:]:
        found = False
        for i in range(len(mergedParams)):
            params = mergedParams[i]
            dist = ellipseDistance(params, ellipse)
            print(dist)
            if dist < dist_threshold:
                mergedParams[i] = sumEllipses(ellipse, mergedParams[i])
                cnt[i] += 1
                found = True
        if not found:
            mergedParams.append(ellipse)
            cnt.append(1)

    valid_merges = []

    for i in range(len(mergedParams)):
        if cnt[i] > qtt_threshold: 
            ell = mergedParams[i]
            center = ell[0]
            axes = ell[1]
            alpha = ell[2]
        
            x, y = center
            d0, d1 = axes

            x /= cnt[i]
            y /= cnt[i]
            d0 /= cnt[i]
            d1 /= cnt[i]       
        
            valid_merges.append([(int(x), int(y)), (int(d0), int(d1)), alpha])
    
    return valid_merges
        
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
def youghongQiangEllipse(image, n_pixels, minDist, maxDist, minLength, maxLength, bags, threshold):
    pixels = np.random.permutation(np.argwhere(image==255))[:n_pixels]
    accLength = maxLength - minLength
    
    ellipses = []
    
    for p1 in pixels:
        for p2 in pixels:
            
            accumulator = np.zeros((bags,))
            p3s = [(0, 0) for _ in range(bags)]
            ds = [0 for _ in range(bags)]
            fs = [0 for _ in range(bags)]
            bs = [0 for _ in range(bags)]
            
            dist, distSq = distance(p1, p2)
            if dist > minDist and dist < maxDist:
                p0    = (p1 + p2) / 2
                a     = dist / 2
                aSq   = distSq / 4

                for p3 in pixels:
                    #verif que p3 n'est ni p1 ni p2...
                    d, dSq = distance(p0, p3)
                    if d > minDist and d < a:
                        f, fSq = distance(p3, p2)
                        cosTau = (aSq + dSq - fSq) / (2 * a * d)
                        cosTauSq = cosTau * cosTau
                        bSq    = ((aSq * dSq) * (1 - cosTauSq)) / (aSq - dSq * cosTauSq)
                        b      = bSq**(1/2)
                        
                        if b > minLength and b < maxLength:
                            bag_id = int((b - minLength) * (bags / accLength))
                            accumulator[bag_id] += 1
                            p3s[bag_id] = p3
                            ds[bag_id] = d
                            fs[bag_id] = f
                            bs[bag_id] = b
                            
                best = np.argmax(accumulator)
                if accumulator[best] > threshold:
                    b = minLength + best * (accLength / bags)   
                    y1, x1 = p1
                    y2, x2 = p2  
                    y0, x0 = p0
                    alpha = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    ellipse = [(int(x0), int(y0)), (int(a), int(b)), alpha] 
                    ellipses.append(ellipse)
                    #print("alpha = {} p0 = {} p1 = {} p2 = {} p3 = {} a = {} d = {} f = {} b' = {} b = {}".format(alpha, p0, p1, p2, p3s[best], a, ds[best], fs[best], b, bs[best]))
    return ellipses
    
if __name__ == "__main__":
    image = cv2.imread("test_img/test_el.png", cv2.IMREAD_GRAYSCALE)
    _, imageBinary = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY)
    
    ellipses1 = youghongQiangEllipse(imageBinary, 300, 10, 150, 10, 150, 360, 4)
    ellipses2 = mergeEllipses(ellipses1, 50, 2)
    
    print(len(ellipses1))
    print(len(ellipses2))
     
    imageBGR = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    for ell in ellipses1:
        cv2.ellipse(imageBGR, ell[0], ell[1], ell[2], 0, 360, (0, 0, 255))
        
    for ell in ellipses2:
        cv2.ellipse(imageBGR, ell[0], ell[1], ell[2], 0, 360, (255, 0, 0))
 
    cv2.imshow("test", imageBGR)
    cv2.waitKey(50000)
