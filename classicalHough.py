import cv2
import numpy as np
from imgTools import display, multiDisplay
from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
from matplotlib import pyplot as plt   # Plot library


if __name__ == "__main__":
	# Load picture, convert to grayscale and detect edges
	image_rgb = data.coffee()[0:220, 160:420]
	image_gray = color.rgb2gray(image_rgb)
	r,g,b = cv2.split(image_rgb)
	image_bgr = cv2.merge( [b,g,r])
	edges = canny(image_gray, sigma=2.0,
		          low_threshold=0.55, high_threshold=0.8)
		          
	# Perform a Hough Transform
	# The accuracy corresponds to the bin size of a major axis.
	# The value is chosen in order to get a single high accumulator.
	# The threshold eliminates low accumulators
	result = hough_ellipse(edges, accuracy=20, threshold=250,
		                   min_size=100, max_size=120)
	result.sort(order='accumulator')

	# Estimated parameters for the ellipse
	best = list(result[-1])
	yc, xc, a, b = [int(round(x)) for x in best[1:5]]
	orientation = best[5]
	img2 = image_rgb*0
	cv2.ellipse(img2, (xc, yc), (a, b), orientation, 0, 3600, 255, 3)
	
	"""
	yc = 200
	xc = 200
	a = 100
	b = 50
	orientation = 45
	img2 = img*0
	cv2.ellipse(img2, (xc, yc), (a, b), orientation, 0, 3600, 255, 3)
	"""

	#display("", img)
	multiDisplay(["", ""], [image_bgr, img2], 2)
