import cv2
import os
import numpy as np

ann = []
f = "test_metrics/ann"
folder =  sorted(os.listdir(f))
for im in folder:
	ann.append(cv2.imread(f+"/"+im, cv2.IMREAD_GRAYSCALE)/255)

det = []
f = "test_metrics/det"
folder =  sorted(os.listdir(f))
for im in folder:
	det.append(cv2.imread(f+"/"+im, cv2.IMREAD_GRAYSCALE)/255)

# Naive metric
def naive(ar1, ar2):
	tmp = abs(ar1-ar2)
	return 1-np.mean(tmp)
	
	
# Metric 1
def metric1(annoted, detected, error_false_neg = 0.2):
	num_to_get = np.sum(annoted)
	num_gotten = np.sum(np.multiply(annoted, detected))
	return min((num_gotten/num_to_get)/(1-error_false_neg), 1)
	

def metric2(annoted, detected, error_false_positive = 0.20):
	detected_false = detected - np.multiply(annoted, detected)
	num_gotten_false = np.sum(detected_false)
	num_to_get = np.sum(annoted)	
	num_not_to_get = detected.shape[0]*detected.shape[1]-num_to_get
	if num_gotten_false == 0:
		m2 = 1
	else:
		m2 = min((num_to_get*error_false_positive)/num_gotten_false, 1)
	return m2

def metric3(annoted, detected, error_false_neg = 0.2, error_false_positive = 0.2):
	m1 = metrics1(annoted, detected, error_false_neg)
	m2 = metrics1(annoted, detected, error_false_positive)
	return (m1+m2)/2
	
val = 0
for i in range(len(det)):
	val += naive(ann[i], det[i])	
val = val/len(det)
print(f'The naive metric is : {val}.')
print(f"The naive metric of {folder[0]} on 'nothing detected' is : {naive(ann[0], (det[0]*0))}")

	
val = 0
for i in range(len(det)):
	val += metric1(ann[i], det[i])
val = val/len(det)
print(f'\nThe metric 1 is : {val}.')
i = 8
print(f"The metric 1  of {folder[i]} on 'detected' is : {metric1(ann[i], (det[i]))}")
print(f"The metric 1  of {folder[i]} on 'nothing detected' is : {metric1(ann[i], (det[i]*0))}")
print(f"The metric 1  of {folder[i]} on 'everything detected' is : {metric1(ann[i], np.ones((det[i].shape[0], det[i].shape[1])))}")

	
val = 0
for i in range(len(det)):
	val += metric2(ann[i], det[i])
val = val/len(det)
print(f'\nThe metric 2 is : {val}.')
i = 8
print(f"The metric 2 of {folder[i]} on 'detected' is : {metric2(ann[i], (det[i]))}")
print(f"The metric 2 of {folder[i]} on 'nothing detected' is : {metric2(ann[i], (det[i]*0))}")
print(f"The metric 2 of {folder[i]} on 'everything detected' is : {metric2(ann[i], np.ones((det[i].shape[0], det[i].shape[1])))}")


val = 0
for i in range(len(det)):
	val += metric2(ann[i], det[i])
val = val/len(det)
print(f'\nThe metric 3 is : {val}.')
i = 8
print(f"The metric 3 of {folder[i]} on 'detected' is : {metric2(ann[i], (det[i]))}")
print(f"The metric 3 of {folder[i]} on 'nothing detected' is : {metric2(ann[i], (det[i]*0))}")
print(f"The metric 3 of {folder[i]} on 'everything detected' is : {metric2(ann[i], np.ones((det[i].shape[0], det[i].shape[1])))}")

