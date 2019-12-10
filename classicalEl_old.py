import cv2
import numpy as np
import math

def intersect(line1, line2):
    a1, b1 = line1
    a2, b2 = line2

    assert a1 != a2

    if a1 == float("inf"):
        return (b1, a2 * b1 + b2)
    if a2 == float("inf"):
        return (b2, a1 * b2 + b1)
        
    x = (b2 - b1) / (a1 - a2)
    y = a1 * x + b1
    return (y, x)

def make_line(point1, point2):
    y1, x1 = point1
    y2, x2 = point2
    
    if x1 == x2:
        return (float("inf"), x1)
    else:
        a = (y1 - y2) / (x1 - x2)
        b = y1 - a * x1
        return (a, b)

def make_kernels(radius, thickness):
    kernels = []
    slopes  = []
    
    kernel_dim = 2 * radius + 1
    for i in range(kernel_dim):
        kernel1 = np.zeros((kernel_dim, kernel_dim), np.float32)
        cv2.line(kernel1, (0, i), (2 * radius, 2 * radius - i), 1, thickness)
        kernel1 = kernel1 / (kernel_dim * kernel_dim)
        kernels.append(kernel1)
        
        slope1, _ = make_line((i, 0), (2 * radius - i, 2 * radius))
        slopes.append(slope1)
        
        if i != 0 and i != 2 * radius:
            kernel2 = np.zeros((kernel_dim, kernel_dim), np.float32)
            cv2.line(kernel2, (i, 0), (2 * radius - i, 2 * radius), 1, thickness)
            kernel2 = kernel2 / (kernel_dim * kernel_dim)
            kernels.append(kernel2)
            
            slope2, _ = make_line((0, i), (2 * radius, 2 * radius - i))
            slopes.append(slope2)
        
    return kernels, slopes
    
def validate(point1, point2, dist_min, dist_max):
    y1, x1 = point1
    y2, x2 = point2    
    
    dist = np.abs(x1 - x2) + np.abs(y1 - y2)
    if(dist < dist_min or dist > dist_max):
        return False

    return True

def compute_tangents(image, kernels, slopes, threshold):
    # Pour chaque point, on sélection la meilleure tangente par filtrage
    cnts = []
    for kernel in kernels:
        cnt = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_ISOLATED)
        cnts.append(cnt)
        
    best = np.argmax(cnts, axis=0)
   
    couples = []
    height, width = image.shape
    for y in range(height):
        for x in range(width):
            tg_index = best[y, x]
            cnt = cnts[best[y, x]]
            
            if cnts[tg_index][y, x] > threshold:
                point = (y, x)
                slope = slopes[tg_index]
                b = y - slope * x if slope != float("inf") else x
                line  = (slope, b)
                couples.append((point, line))
    return couples
    
def dist(point1, point2):
    y1, x1 = point1
    y2, x2 = point2
    
    return ((x1 - x2)**2 + (y1 - y2)**2)**(1/2)
    
def sort_triplet(triplet):
    triplet_new = []
    pt0, tg0 = triplet[0]
    pt1, tg1 = triplet[1]
    pt2, tg2 = triplet[2]
    
    d01 = dist(pt0, pt1)
    d02 = dist(pt0, pt2)
    d12 = dist(pt1, pt2)

    if d01 <= d02 and d01 <= d12:
        pivot = pt2
        tg_pivot = tg2
        tmp_pt0 = pt0
        tmp_tg0 = tg0
        tmp_pt1 = pt1
        tmp_tg1 = tg1
        d0 = d02
        d1 = d12
    elif d02 <= d01 and d02 <= d12:
        pivot = pt1
        tg_pivot = tg1
        tmp_pt0 = pt0
        tmp_tg0 = tg0
        tmp_pt1 = pt2
        tmp_tg1 = tg2
        d0 = d01
        d1 = d12
    elif d12 <= d01 and d12 <= d02:
        pivot = pt0
        tg_pivot = tg0
        tmp_pt0 = pt1
        tmp_tg0 = tg1
        tmp_pt1 = pt2
        tmp_tg1 = tg2
        d0 = d01
        d1 = d02
        
    return [(pivot, tg_pivot), (tmp_pt0, tmp_tg0), (tmp_pt1, tmp_tg1)], d0, d1
        
def compute_centers(couples, min_dist, max_dist, min_tg_dist):
    triplets = [couples[x:x+3] for x in range(0, len(couples), 3)]
    
    centers = []

    for triplet in triplets:
        if len(triplet) == 3:
        
            tiplet, d0, d1 = sort_triplet(triplet)  
                   
            if d0 >= min_dist and d1 >= min_dist and d0 < max_dist and d1 < max_dist:
                pt0, (a0, b0) = triplet[0]
                pt1, (a1, b1) = triplet[1]
                pt2, (a2, b2) = triplet[2]
                
                if np.abs(a0 - a1) > min_tg_dist and np.abs(a0 - a2) > min_tg_dist:
                    m01 = (1/2) * (pt0 + pt1)
                    t01 = intersect((a0, b0), (a1, b1))
                    (a01, b01) = make_line(m01, t01)
                    
                    m02 = (1/2) * (pt0 + pt2)
                    t02 = intersect((a0, b0), (a2, b2))
                    (a02, b02) = make_line(m02, t02)
                
                    if np.abs(a01 - a02) > min_tg_dist:
                        centers.append(intersect((a01, b01), (a02, b02)))
    return centers
    
def make_hough2(image, threshold, radius=10, thickness=3, percentage=1, min_dist=5, max_dist=200, min_tg_dist=0):
    kernels, slopes = make_kernels(radius, thickness)
    couples = compute_tangents(image, kernels, slopes, threshold)
    selected = np.random.permutation(couples)[:int(percentage * len(couples))]
    print("{} {}".format(len(couples), len(selected)))
    centers = compute_centers(selected, min_dist, max_dist, min_tg_dist)
    
    for center in centers:
        y, x = center
        cv2.circle(image, (int(x), int(y)), 3, 200)    
    
    print(len(centers))

def make_hough(couples, width, height, line_length, dist_min=20, dist_max=80):
    rejected = 0
    cnt = 0
    space = np.zeros((width, height))
    length = len(couples)
    
    # Chercher les droites liées à chaque couple de couples
    lines = []
    for i in range(length):
        for j in range(i):     
            cnt += 1   
            # si les conditions ne sont pas remplies, on skip cette itération
           
            point1, tangent1 = couples[i]
            point2, tangent2 = couples[j]
            
            a1, _ = tangent1
            a2, _ = tangent2
            
            if np.abs(a1 - a2) > 1:
                # centre entre les deux points            
                pc = (1/2) * (point1 + point2)
                            
                # croisement des deux tangentes

                pt = intersect(tangent1, tangent2)
                
                if validate(pc, pt, dist_min, dist_max):
                    # a et b de la droite passant par C et T
                    #current = make_line(pc, pt)
                    a, b = make_line(pc, pt)
                    
                    # la ligne courrante est comparée aux lignes précédentes
                    #test = range(max(0, int(pc[1]) - line_length), min(width, int(pc[1]) + line_length), 1)
                    for x in range(width):
                        y = a * x + b
                        if(y >= 0 and y < height):
                            space[int(y), x] += 1
                            #space[int(y), x] = 255
                    """current = (a, b)
                    for line in lines:
                        if current[0] != line[0]:
                            y, x = intersect(current, line)
                            
                            if x >= 0 and x < width and y >= 0 and y < height: 
                                space[int(y), int(x)] = 255
                        
                    # la ligne courrante est ajoutée aux autres lignes
                    lines.append(current)"""
                else:
                    rejected += 1
            else:
                rejected += 1
    print("REJECTED_CNT: {} / {}".format(rejected, cnt))
    return space
    
def draw_lines(couples, width, height, line_length):
    img = np.zeros((width, height))
    for couple in couples:
        point, line = couple
        y, x = point
        a, b = line
        test = range(max(0, int(x) - line_length), min(width, int(x) + line_length), 1)
        for x in test:
            y = a * x + b
            if y >= 0 and y < height:
                img[int(y), x] = 255
    cv2.imshow("Draw", img)    
            
if __name__ == "__main__":

    image = cv2.imread("test_img/test_el.png", cv2.IMREAD_GRAYSCALE)
    _, image = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY)

    radius = 10
    threshold = 255 / (2 * radius + 1)
    print("Threshold = {}".format(threshold))  
    make_hough2(image, threshold, radius=10, thickness=3, )
    cv2.imshow("image", image)
    cv2.waitKey(500000)
    """
    # On crée les masques pour les tangentes
    kernels, slopes = make_kernels(10, 3)
    
    # Pour chaque point, on sélection la meilleure tangente par filtrage
    cnts = []
    for kernel in kernels:
        # TODO mettre le facteur directement dans la création de kernels
        # Solution : diviser les kernels par leur aire totale
        cnt = cv2.filter2D(image, -1, (1/255) * kernel, borderType=cv2.BORDER_ISOLATED)
        cnts.append(cnt)
        
    best = np.argmax(cnts, axis=0)

    # On va chercher après les points blancs dans l'image et évaluer la
    # tangente en ces points
    couples = []
    # TODO ajouter en parametre la taille du sample
    points = np.random.permutation(np.argwhere(image==255))[:100]
    for point in points:
        tangent_index = best[point[0], point[1]]
        y, x  = point
        slope = slopes[tangent_index]
        b = y - slope * x if slope != float("inf") else x
        line  = (slope, b) 
        couples.append((point, line))

    draw_lines(couples, 500, 500, 50)
    hough_space = make_hough(couples, 500, 500, 30)

    cv2.imshow("hough", hough_space/10)
    print(np.argmax(hough_space))
    cv2.waitKey(500000)
    """
