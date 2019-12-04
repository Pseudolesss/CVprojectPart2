# Evaluation metrics
## Ellipses
### Distance between two ellipses
It is assumed that the ellipses are defined by 5 parameters:
- Xc : the vert. coord of the center of the ellipse
- Yc : the horiz. coord of the center of the ellipse
- theta : the angle of the main axis
- a  : half the length of main axis
- b  : half the length of sub axis

~~~
elps1 = [Xc1, Yc1, theta1, a1, b1]
elps2 = [Xc2, Yc2, theta2, a2, b2]
W = [Wx, Wy, Wt, Wa, Wb]
dist = Wx * abs(Xc1-Xc2) + Wy * abs(Yc1-Yc2) + Wt * abs(theta1-theta2) + Wa * abs(a1-a2) + Wb * abs(b1-b2)
~~~
The weight W is tuned such that the maximal acceptable error on each parameters will lead to an increase in the distance of 1. Thus if we want an acceptable error of theta of 15 degrees, we fix Wt = 1/15. This way, we can fix the tresh_dist = 5. This is the maximal acceptable distance between two ellipses. By fixing it at 5, we only need to tune W. The parameters W will depend on the precision expected on the labelled ellipse. (If we assume that the precision of labelling for the center is not lower than 2, then we can put Xc = 2 and Yc = 2).

### Evaluation on multiple images
Let ground_truth and detected both being a list of list of ellipses, respectivelly being the labeled ellipses for each images and the detected ellipses for each images. The evaluation is done on the following pseudo-code:

~~~
metrics_elps(ground_truth, detected, tresh_dist):
    eval = 0
    for each image:
        gt_elps = ground_truth(image)
        dt_elps = detected(image)

        num = 0
        denom = max(len(gt_elps), len(dt_elps))

        dists = np.zeros((len(gt_elps), len(dt_elps)))
        for el in dists:
            el = dist(gt_elps[index1(el)], dt_elps[index2(el)]

        index1_rem = []
        index2_rem = []
        for i in range(min(len(gt_elps), len(dt_elps))):
            el = min(dists) where (not index1(el) in index1_rem) and (not index2(el) in index2_rem)
            index1_rem.append(index1(el))
            index2_rem.append(index2(el))

            num += min(1, tresh_dist/el)

        eval += num/denom
    return eval/num_image
~~~

In other words, the evaluation metric give a value between 0 and 1, 1 being the best accuracy. The evaluation for multiple images is the mean of all image evaluations. For each image, the evaluation of this images is a fraction. Let
![Function f metrics](https://github.com/Pseudolesss/CVprojectPart2/blob/master/Documentations/gifs/metrics_f.gif "Function f metrics")
, a function which returns 1 if x < tresh_dist and something < 1 in other cases. The choice of the tresh_dist is done such that a distance lower than it doesn't make sense compared to the precision of the labelling.

The numerator is the sum of function f on the smallest distances between the detected ellipses and the labeled ellipses where each detected and labeled ellipse can only be used once. When the number of detected and labeled ellipses aren't the same, the distance are computed until no pair can be made.

In the other hand, the denominator is the maximum between the number of detected and labeled ellipses. Thus the value can decrease due to two reasons : if the detection detect less or more ellipses than in the ground truth and if the detected ellipses are distant from the labeled ones.
