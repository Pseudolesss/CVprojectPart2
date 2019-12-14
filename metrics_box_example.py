import cv2                             # OpenCV
from imgTools import display, multiDisplay
import numpy as np
from metrics_box import metric_box

def evaluate():
	elgd = [np.array([50, 100, 150, 50], dtype=np.float32)]
	          # the values are Xc, Yc (the coord. of center), theta (angle of main axis in 
		      # degree), a (half length of main axis), b ((half length of sub axis)
	elsd = []
	elsd.append([np.array([50, 100, 150, 50], dtype=np.float32)])
	elsd.append([np.array([60, 80, 170, 30], dtype=np.float32), np.array([60, 120, 150, 70], dtype=np.float32)])
	elsd.append([np.array([52, 98, 148, 52], dtype=np.float32)])
	elsd.append([np.array([60, 130, 80, 70], dtype=np.float32)])
	elsd.append([np.array([60, 130, 80, 70], dtype=np.float32), 
	             np.array([50, 100, 150, 50], dtype=np.float32), 
	             np.array([60, 120, 150, 70], dtype=np.float32)])
	
	imgs = []
	img1 = np.zeros((300, 200, 3))
	for i in range(len(elsd)):
		imgs.append(np.zeros((300, 200, 3)))
	
	for ell in elgd:
		ldc = (ell[0], ell[1])
		luc = (ell[0], ell[3])
		ruc = (ell[2], ell[3])
		rdc = (ell[2], ell[1])
		cv2.line(img1, ldc, luc, (255, 255, 255), 2)
		cv2.line(img1, luc, ruc, (255, 255, 255), 2)
		cv2.line(img1, ruc, rdc, (255, 255, 255), 2)
		cv2.line(img1, rdc, ldc, (255, 255, 255), 2)
		for i in range(len(imgs)):
			cv2.line(imgs[i], ldc, luc, (255, 255, 255), 2)
			cv2.line(imgs[i], luc, ruc, (255, 255, 255), 2)
			cv2.line(imgs[i], ruc, rdc, (255, 255, 255), 2)
			cv2.line(imgs[i], rdc, ldc, (255, 255, 255), 2)
	
	print("\nStart evaluation...")
	mn = 0
	m1 = 0
	to_display_metrics = []
	for i in range(len(elsd)) :			
		# detect ellipse
		eld = elsd[i]		
		
		# metric
		m1_i, m2_i = metric_box([elgd], [eld])
		to_display_metrics.append([m1_i, m2_i])
		
		# apply ellipse on image
		for ell in eld:
			ldc = (ell[0], ell[1])
			luc = (ell[2], ell[1])
			ruc = (ell[2], ell[3])
			rdc = (ell[0], ell[3])
			cv2.line(imgs[i], ldc, luc, (0, 0, 255), 2)
			cv2.line(imgs[i], luc, ruc, (0, 0, 255), 2)
			cv2.line(imgs[i], ruc, rdc, (0, 0, 255), 2)
			cv2.line(imgs[i], rdc, ldc, (0, 0, 255), 2)
	
	print(f"\n\nThe ground truth ellipse is : \n\tXldc = {elgd[0][0]}, Yldc = {elgd[0][1]}, Xruc = {elgd[0][2]},"+\
		  f" Yruc = {elgd[0][3]}.")	
	txt = ['Ground truth']
	for i in range(len(to_display_metrics)):
		print('\n--------------------------------')
		print(f'The metrics for the ellipse(s) :')
		for ell in elsd[i]:
			print(f'\tXldc = {ell[0]}, Yldc = {ell[1]}, '+\
				  f'Xruc = {ell[2]}, Yruc = {ell[3]}.')
		print(f"Penalizing metric =     {to_display_metrics[i][0]}")
		print(f"Non penalizing metric = {to_display_metrics[i][1]}")
		txt.append(f'metric = {to_display_metrics[i][0]}')
	
	multiDisplay(txt, [img1]+imgs, 3)

if __name__ == '__main__':
	evaluate()
