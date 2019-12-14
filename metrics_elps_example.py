import cv2                             # OpenCV
from imgTools import display, multiDisplay
import numpy as np
from metrics import metric_elps

def evaluate():
	elgd = [np.array([150, 100, 45, 40, 20], dtype=np.float32)]
	          # the values are Xc, Yc (the coord. of center), theta (angle of main axis in 
		      # degree), a (half length of main axis), b ((half length of sub axis)
	elsd = []
	elsd.append([np.array([150, 100, 45+180, 40, 20], dtype=np.float32)])
	elsd.append([np.array([80, 100, 45, 40, 20], dtype=np.float32), np.array([80, 40, 6, 40, 20], dtype=np.float32)])
	elsd.append([np.array([148, 102, 45+5, 43, 17], dtype=np.float32)])
	elsd.append([np.array([138, 112, 45+10, 40, 20], dtype=np.float32)])
	elsd.append([np.array([150, 100, 45, 40, 20], dtype=np.float32), 
	             np.array([40, 1, 45+180, 40, 20], dtype=np.float32), 
	             np.array([50, 140, 40, 40, 30], dtype=np.float32), 
	             np.array([200, 200, 45, 80, 20], dtype=np.float32)])
	
	imgs = []
	img1 = np.zeros((300, 200, 3))
	for i in range(len(elsd)):
		imgs.append(np.zeros((300, 200, 3)))
	
	for ell in elgd:
		cv2.ellipse(img1, (ell[0], ell[1]), (ell[3], ell[4]), ell[2], 0, 360, (255, 255, 255))
		for i in range(len(imgs)):
			cv2.ellipse(imgs[i], (ell[0], ell[1]), (ell[3], ell[4]), ell[2], 0, 360, (255, 255, 255))
	
	print("\nStart evaluation...")
	mn = 0
	m1 = 0
	to_display_metrics = []
	for i in range(len(elsd)) :			
		# detect ellipse
		eld = elsd[i]		
		
		# metric
		m1_i, m2_i = metric_elps([elgd], [eld])
		to_display_metrics.append([m1_i, m2_i])
		
		# apply ellipse on image
		for ell in eld:
			cv2.ellipse(imgs[i], (int(ell[0]), int(ell[1])), (ell[3], ell[4]), ell[2], 0, 360, (0, 0, 255))
	
	print(f"\n\nThe ground truth ellipse is : \n\tXc = {elgd[0][0]}, Yc = {elgd[0][1]}, theta = {elgd[0][2]},"+\
		  f" a = {elgd[0][3]}, b = {elgd[0][4]}.")	
	txt = ['Ground truth']
	for i in range(len(to_display_metrics)):
		print('\n--------------------------------')
		print(f'The metrics for the ellipse(s) :')
		for ell in elsd[i]:
			print(f'\tXc = {ell[0]}, Yc = {ell[1]}, '+\
				  f'theta = {ell[2]}, a = {ell[3]}, b = {ell[4]}.')
		print(f"Penalizing metric =     {to_display_metrics[i][0]}")
		print(f"Non penalizing metric = {to_display_metrics[i][1]}")
		txt.append(f'metric = {to_display_metrics[i][0]}')
	
	multiDisplay(txt, [img1]+imgs, 3)

if __name__ == '__main__':
	evaluate()
