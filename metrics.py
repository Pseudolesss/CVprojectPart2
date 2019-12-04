import numpy as np

def dist(elps1, elps2):
	"""
	Give the distance between the two ellipses
	@Args:
		elps1:
		elps2:
	@Return
		[float] distance between the two ellipses
	"""
	return abs(elps1-elps2)

def metric_elps(ground_truth, detected):
	"""
	Evaluate the detection of ellipse compared to the ground truth.
	@Args:
		ground_truth:		[list of list of elps object] the ellipses labeled for each image
		detected:			[list of list of elps object] the ellipses detected for each image
	@Return:
		[float] evaluation of the detection method between 0 (worst) and 1 (best). (-1 if error)
	"""
	debug = True
	l1 = len(ground_truth)
	l2 = len(detected)
	
	if not l1 == l2:
		return -1
		
	eval = 0
	tresh_dist = 2 # distance below which more precision doesn't make sense (label not that precise)
		
	for i in range(l1):
		l1i = len(ground_truth[i])
		l2i = len(detected[i])
		num = 0
		denom = max(l1i, l2i) # denom increases if too many elps detected compared to the truth

		# compute all distances
		dists = np.zeros((l1i, l2i)) 
		for j in range(l1i):
			for k in range(l2i):
				dists[j,k] = dist(ground_truth[i][j], detected[i][k])

		# find clostest to ground truth
		for j in range(min(l1i, l2i)):
			if j > 0:
				if debug:
					print(f"Min = dists[{ind1},{ind2}] = {dists[ind1, ind2]}")
				# remove line of elps ground truth used and elps detected used
				dists = np.delete(dists, ind2, 1)
				dists = np.delete(dists, ind1, 0)
			if debug:
				print(dists)
			argmin = np.argmin(dists)
			ind2 = argmin%dists.shape[1]
			ind1 = int((argmin-ind2)/dists.shape[1])
			
			delta = 0.00001
			num += min(1, (tresh_dist+delta) / (dists[ind1, ind2]+delta))
			
		if debug:
			print(f"Min = dists[{ind1},{ind2}] = {dists[ind1, ind2]}")
		eval += num/denom
	return eval/l1
			
			
if __name__ == "__main__":
	gt = [[1,3.5]]
	det = [[4,3.5,5]]
	print(f'Final metrics = {metric_elps(gt, det)}')
