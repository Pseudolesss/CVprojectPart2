import numpy as np


def metricNaive(ar1, ar2):
	"""
	Naive metrics to compare the dectected segments and the annoted one.
	@args:
		ar1:	[np array] image with seg detected in greyscale
		ar2:	[np array] image with seg annoted in greyscale
	@return
		[float] score of detection : 0 == worst accuracy and perfect accuracy
	"""
	tmp = abs(ar1-ar2)
	return 1-np.mean(tmp)
	

def metricFalseNeg(annoted, detected, error_false_neg = 0.2):
	"""
	Metrics to compare the dectected segments and the annoted one, based on the false negative (segment not
	detected).
	@args:
		annoted:	[np array] image with seg annoted in greyscale
		detected:	[np array] image with seg detected in greyscale
		error_false_neg : 	[float] acceptable error for false negative. (Accuracy > 1-error will have score of 1)
	@return
		[float] score of detection : 0 == worst accuracy and perfect accuracy
	"""
	num_to_get = np.sum(annoted)
	num_gotten = np.sum(np.multiply(annoted, detected))
	return min((num_gotten/num_to_get)/(1-error_false_neg), 1)
	

def metricFalsePos(annoted, detected, error_false_positive = 0.20):
	"""
	Metrics to compare the dectected segments and the annoted one, based on the false positive (segment	detected
	inexistant).
	@args:
		annoted:	[np array] image with seg annoted in greyscale
		detected:	[np array] image with seg detected in greyscale
		error_false_positive : 	[float] acceptable error for false positive.
	@return
		[float] score of detection : 0 == worst score and perfect score
	"""
	detected_false = detected - np.multiply(annoted, detected)
	num_gotten_false = np.sum(detected_false)
	num_to_get = np.sum(annoted)	
	num_not_to_get = detected.shape[0]*detected.shape[1]-num_to_get
	if num_gotten_false == 0:
		m2 = 1
	else:
		m2 = min((num_to_get*error_false_positive)/num_gotten_false, 1)
	return m2

def metricTot(annoted, detected, error_false_neg = 0.2, error_false_positive = 0.2):
	"""
	Metrics to compare the dectected segments and the annoted one, based on the false positive and false negative.
	inexistant).
	@args:
		annoted:	[np array] image with seg annoted in greyscale
		detected:	[np array] image with seg detected in greyscale
		error_false_neg : 	[float] acceptable error for false negative.
		error_false_positive : 	[float] acceptable error for false positive.
	@return
		[float] score of detection : 0 == worst score and perfect score
	"""
	m1 = metricFalseNeg(annoted, detected, error_false_neg)
	m2 = metricFalsePos(annoted, detected, error_false_positive)
	return (m1+m2)/2

