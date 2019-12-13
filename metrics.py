import numpy as np
import keras.backend as k


def dist(elps1, elps2, weight=[1 / 2, 1 / 2, 1 / 5, 1 / 3, 1 / 3]):
    """
	Give the distance between the two ellipses
	@Args:
		elps1:		[np array of 5 values] the values are Xc, Yc (the coord. of center), theta (angle of main axis in 
		            degree), a (half length of main axis), b ((half length of sub axis).
		elps2:		[np array of 5 values] the values are Xc, Yc (the coord. of center), theta (angle of main axis in 
		            degree), a (half length of main axis), b ((half length of sub axis).
		weight:		[np array of 5 values] how much error is accepted for each parameters. (eg. for the weight = [1/2,
		            1/2, 1/5, 1/3, 1/3], this means, an error of 2 pixels or lower is OK for Xc and Yc, an error of 5 
		            degrees or less is OK for theta and an error of 3 pixels or more is OK for the axis. Those cumulated
		            errors are OK.)
	@Return
		[float] distance between the two ellipses
	"""

    # Use keras instead of numpy in order to avoid symbolic / non symbolic conflicts in the custom loss
    diff = k.abs(elps1 - elps2)
    diff = diff * weight
    diff = diff / k.sum(weight)
    diff = k.sum(diff)

    # diff considering a 180degree change in angle
    diff2 = k.abs(k.abs(elps1 - elps2)- np.array([0, 0, 180, 0, 0], dtype=np.float32))  # array OK for keras?
    diff2 = diff2 * weight
    diff2 = diff2 / k.sum(weight)
    diff2 = k.sum(diff2)

    return min(diff, diff2) # min OK for keras?


def metric_elps(ground_truth, detected, weight=[1 / 2, 1 / 2, 1 / 5, 1 / 3, 1 / 3]):
    """
	Evaluate the detection of ellipse compared to the ground truth.
	@Args:
		ground_truth:		[list of list of elps object] the ellipses labeled for each image
		detected:			[list of list of elps object] the ellipses detected for each image
		weight:				[np array of 5 values] how much error is accepted for each parameters. (see fct dist)
	@Return:
		eval1 [float] evaluation of the detection method between 0 (worst) and 1 (best). (-1 if error)
		eval2 [float] evaluation of the detection method between 0 (worst) and 1 (best). (-1 if error). In this 
		metrics, there are no penalization if there is more detected ellipse than existing ones in the ground truth. 
	"""
    debug = False
    l1 = len(ground_truth)
    l2 = len(detected)

    if not l1 == l2:
        return -1, -1

    eval1 = 0
    eval2 = 0
    tresh_dist = 5  # distance below which more precision doesn't make sense (label not that precise)
    # Value is 5 because it is assumed that the weight of dist() has been tuned such that the max
    # accepted error for each of the 5 parameters would be equal to 1. Thus 1*5 = 5.

    for i in range(l1):
        l1i = len(ground_truth[i])
        l2i = len(detected[i])
        num = 0

        if l1i == 0:
            if l2i == 0:
                eval1 += 1
                eval2 += 1
        else:
            denom1 = max(l1i, l2i)  # denom increases if too many elps detected compared to the truth
            denom2 = l1i # non penalizing denominator

            # compute all distances
            dists = np.zeros((l1i, l2i))
            for j in range(l1i):
                for k in range(l2i):
                    dists[j, k] = dist(ground_truth[i][j], detected[i][k], weight)

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
                ind2 = argmin % dists.shape[1]
                ind1 = int((argmin - ind2) / dists.shape[1])

                delta = 0.00001
                num += min(1, (tresh_dist + delta) / (dists[ind1, ind2] + delta))
            if debug:
                print(f"Min = dists[{ind1},{ind2}] = {dists[ind1, ind2]}")
            eval1 += num / denom1
            eval2 += num / denom2
    return eval1 / l1, eval2 / l1


if __name__ == "__main__":
    gt = [[np.array([1, 3.5, 2, 2, 1], dtype=np.float32), np.array([4, 2.5, 2, 7, 1], dtype=np.float32), 
    np.array([1, 3.5, 7, 2, 8], dtype=np.float32)]]
    dt = [[np.array([4, 5.5, 5, 2, 1], dtype=np.float32), np.array([8, 3.5, 15, 20, 1], dtype=np.float32),
    np.array([3.5, 5.5, 13, 2, 8], dtype=np.float32)]]
    print(f'Final metrics = {metric_elps(gt, dt)}')
