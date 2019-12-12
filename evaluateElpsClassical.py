import cv2
import csv
import os
import sys
import numpy as np
from pathlib import Path
from imgTools import display, multiDisplay
from metrics import metric_elps
from classical import youghongQiangEllipse


database_directory = os.path.join(os.getcwd(), 'images_database')
annotationFile = os.path.join(database_directory, 'CV2019_Annots.csv')


def extract_annotations_eye():
    images_annotations = dict()
    with open(annotationFile, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        result = []

        for row in csv_reader:
            # print(row)
            # print(row[0])
            if 'elps_eye' in row[0]:
                result.append([row[1:], row[0]])

        for eye, image_name in result:

            # We assume only one notation
            tmp = np.array(eye[1:], dtype=np.float32)
            points = np.reshape(tmp, (-1, 2))
            # Convert cartesian y to open cv y coordinate
            for elem in points:
                elem[1] = 240 - elem[1]

            ellipse = cv2.fitEllipse(points)

            # We assume only one notation
            center = (int(ellipse[0][0]), int(ellipse[0][1]))
            size = (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2))
            angle = int(ellipse[2])

            images_annotations[image_name] = [center[0], center[1], angle,
                                              size[0], size[1]]

        return images_annotations


def get_model_data_eye_ellipse():
    image_names = []
    images_list = []
    result = list(Path("images_database/eyes/preprocess/").glob('*.png'))
    for file in result:  # fileName
        image = cv2.imread(str(file.resolve()), cv2.IMREAD_GRAYSCALE)
        images_list.append(image)
        image_names.append(file.name)

    annotations_dict = extract_annotations_eye()

    annotations_list = []
    for image_name in image_names:
        if image_name in annotations_dict:
            annotations_list.append(annotations_dict[image_name])
        else:
            annotations_list.append([])

    return images_list, annotations_list, annotations_dict


def get_model_data_eye_no_ellipse():
    images_list = []
    result = list(Path("images_database/eyes/noEllipses/partial/").glob(
        'noelps_eye*'))

    for file in result:  # fileName
        images_list.append(
            cv2.imread(str(file.resolve()), cv2.IMREAD_GRAYSCALE))

    return images_list

def detect_elps_classical(img):
	"""
	Detect elps in image
	@Args:
		img:		[np. array] image
	@Return:
		[list of numpy array in float32] A list of the ellipse detected with for each ellipse an array : 
		Xc, Yc (the coord. of center), theta (angle of main axis in degree), a (half length of main axis), b ((half
		length of sub axis).
	"""	
	# Detect ellipses
	elps = youghongQiangEllipse(imageBinary, 15, 50, 5, 25, 40, 7, 0.5)
	return [np.array([el[0][0], el[0][1], el[2], el[1][0], el[1][1]], dtype=np.float32) for el in elps]

if __name__ == '__main__':
	images_list, annotations_list, annotations_dict = get_model_data_eye_ellipse()
	images_list_no_elps = get_model_data_eye_no_ellipse()
	
	print(len(images_list))
	print(len(annotations_list))
	
	m1 = 0
	m2 = 0
	lim = len(images_list)
	sys.stdout.write("\rProgress : 0% ")
	sys.stdout.flush()
	for i in range(lim):
		sys.stdout.write(
			"\rProgress : %.2f%% \t - Value m1 = %.2f \t m2 = %.2f    " %((i+1) / lim * 100, m1 , m2))
		sys.stdout.flush()
		img = images_list[i]
		#img = cv2.resize(img,(360, 200)) # Images resized to 360p
		img = cv2.Canny(img, 20, 100)
		ann = [[np.array(annotations_list[i], dtype=np.float32)]]
		det = [detect_elps_classical(img)]
		
		m1i, m2i = metric_elps([ann], det, [1 / 2, 1 / 2, 1 / 10, 1 / 3, 1 / 3])
		m1 = (m1*i + m1i)/(i+1)
		m2 = (m2*i + m2i)/(i+1)
		
		"""
		img2 = img.copy()*0
		img3 = img.copy()*0
		for ell in det[0]:
			cv2.ellipse(img2, (ell[0], ell[1]), (ell[3], ell[4]), ell[2], 0, 360, (255))
		for ell in ann[0]:
			cv2.ellipse(img3, (ell[0], ell[1]), (ell[3], ell[4]), ell[2], 0, 360, (255))
		img4 = cv2.merge([img*0, img2, img3])
		print(f"\n\nNumber of ellipses detected : {len(det[0])}")
		multiDisplay(['Img provided','Ellipse detected', 'True ellipse', 'All'], [img, img2, img3, img4], 2, height=7)
		"""
	m1 = m1
	m2 = m2
	
	print(f"The metrics are : \nmetricElps_default = {m1}\nmetricElps_notPenalized = {m2}")
