import cv2
import csv
import os
import numpy as np
from pathlib import Path

import cv2
import csv
import os
import numpy as np
from pathlib import Path
from imgTools import display, multiDisplay
import matplotlib.pyplot as plt
from metrics import metric_elps
from classical import youghongQiangEllipse
from imgTools import display, multiDisplay
import matplotlib.pyplot as plt
import sys
from images_database.preprocess_eyes import img_eye_partial_preprocessing

database_directory = os.path.join(os.getcwd(), 'ReportImages')
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


def get_model_data_eye_ellipse(folders = ["images_database/eyes/preprocess/"]):
	image_names = []
	images_list = []

	all_path = []
	all_names = []
	for i in range(len(folders)):
		all_names += os.listdir(folders[i])
		all_files_in_folder = os.listdir(folders[i])
		for fil in all_files_in_folder:
			all_path.append(folders[i]+"/"+fil)
	lines_img_names = sorted([img for img in all_names if ("elps_eye" in img) and not ("noelps_eye" in img) 
		                      and img.endswith(".png")])
	lines_img_path = sorted([img for img in all_path if ("elps_eye" in img) and not ("noelps_eye" in img) 
		                     and img.endswith(".png")])
		                      
	for path in lines_img_path:
		image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		image = img_eye_partial_preprocessing(image)
		images_list.append(image)
		image_names.append(path.split("/")[-1])

	annotations_dict = extract_annotations_eye()

	annotations_list = []
	for image_name in image_names:
		if image_name in annotations_dict:
		    annotations_list.append(annotations_dict[image_name])
		else:
		    annotations_list.append([])

	return images_list, annotations_list, annotations_dict


def get_model_data_eye_no_ellipse(folders = ["images_database/eyes/noEllipses/full/"]):
	images_list = []

	all_path = []
	all_names = []
	for i in range(len(folders)):
		all_names += os.listdir(folders[i])
		all_files_in_folder = os.listdir(folders[i])
		for fil in all_files_in_folder:
			all_path.append(folders[i]+"/"+fil)
	lines_img_names = sorted([img for img in all_names if ("noelps_eye" in img) and img.endswith(".png")])
	lines_img_path = sorted([img for img in all_path if ("noelps_eye" in img) and img.endswith(".png")])

	for img_path in lines_img_path:  # fileName
		image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
		image = img_eye_partial_preprocessing(image)
		images_list.append(image)

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
	# Preprocess
	#_, imageBinary = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
	imageBinary = img
	
	# Detect ellipses
	elps = youghongQiangEllipse(imageBinary, 15, 50, 5, 25, 40, 7, 0.5)
	return [np.array([el[0][0], el[0][1], el[2], el[1][0], el[1][1]], dtype=np.float32) for el in elps]

def main(folders = None):
	if folders is None:
		folders = ["images_database/eyes/noEllipses/full/", "images_database/eyes/preprocess/"]
	
	# Lines images
	resp = input(f'\nThe folders containing the images are "\n{folders}\n". Do you want to change them? [y, n] : ')
	if resp == "y":
		folders = []
		while True:
			path = input("What is the new path to a folder containing images? : ")
			folders.append(path)
			resp = input(f'The folder containing the images are "\n{folders}\n". Is there more? [y, n] : ')
			if resp == "n":
				break
				
	images_list, annotations_list, annotations_dict = get_model_data_eye_ellipse(folders)
	images_list_no_elps = get_model_data_eye_no_ellipse(folders)
	
	m1 = 0
	m2 = 0
	lim = len(images_list)
	sys.stdout.write("\rProgress : 0% ")
	sys.stdout.flush()
	cnt = 0
	for i in range(lim):
		sys.stdout.write(
			"\rProgress : %.2f%% \t - Value m1 = %.2f \t m2 = %.2f    " %((i+1) / lim * 100, m1 , m2))
		sys.stdout.flush()
		img = images_list[i]
		img = cv2.Canny(img, 20, 100)
		ann = [[np.array(annotations_list[i], dtype=np.float32)]]
		det = [detect_elps_classical(img)]
		
		m1i, m2i = metric_elps([ann], det, [1 / 2, 1 / 2, 1 / 10, 1 / 3, 1 / 3])
		m1 = (m1*cnt + m1i)/(cnt+1)
		m2 = (m2*cnt + m2i)/(cnt+1)
		cnt += 1
		
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

if __name__ == '__main__':
	main()
