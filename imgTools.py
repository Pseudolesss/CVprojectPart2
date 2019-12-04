import numpy as np                     # Numerical algorithms on arrays
import cv2                             # OpenCV
from matplotlib import pyplot as plt   # Plot library

def display(title, img, width=15, height=15):
    """ 
    Display the image given.
    
    @Args:
        title:    [str] title to display
        img:      [np array] Image to display which can be in BGR ([? x ? x 3]) or gray ([? x ?])
        width      [int] width of figure
        heigth     [int] height of figure
    """
    if img.shape[-1]==3: # BGR to RGB
        b,g,r = cv2.split(img)
        imgRgb = cv2.merge( [r,g,b])
    else: # Gray to RGB
        imgRgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Display using pyplot instead of cv2 because it might cause the jupyter notebook to crash.
    plt.figure(figsize = (width, height))
    fig = plt.imshow(imgRgb.astype(np.uint8))
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.title(title)
    plt.show()

def multiDisplay(titles, imgs, imgPerLine, width=15, height=15):
    """ 
    Display the images given.
    
    @Args:
        titles:    [list of str] titles to display for each image
        imgs:      [list of np array] Array of images to display which can be in BGR ([? x ? x 3]) or gray ([? x ?])
        imgPerLine [int] the number of images to display per lines
        width      [int] width of figure
        heigth     [int] height of figure
    """
    length = len(titles)
    numLines = int((length-length%imgPerLine)/imgPerLine)
    if length%imgPerLine > 0 :
        numLines += 1
    fig = plt.figure(figsize = (width, height))
    tot = 0
    for i in range(numLines):
        for j in range(imgPerLine):
            fig.add_subplot(numLines, imgPerLine, tot+1)
            
            if imgs[tot].shape[-1]==3: # BGR to RGB
                b,g,r = cv2.split(imgs[tot])
                imgRgb = cv2.merge( [r,g,b])
            else: # Gray to RGB
                imgRgb = cv2.cvtColor(imgs[tot], cv2.COLOR_GRAY2RGB)
                    
            plt.imshow(imgRgb.astype(np.uint8))
            plt.title(titles[tot])
            fig.axes[tot].get_xaxis().set_visible(False)
            fig.axes[tot].get_yaxis().set_visible(False)
            
            tot += 1
            if tot == length:
                break
        if tot == length:
            break
            
    plt.show()
