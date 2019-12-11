from cytomine import Cytomine
from cytomine.models import ProjectCollection, ImageInstanceCollection, AnnotationCollection
import getpass
import shapely
from shapely import wkt
from shapely.affinity import affine_transform
import cv2                             # OpenCV
from imgTools import display, multiDisplay
import os, sys
import logging
import segment_detector as sd
import numpy as np

def evaluate():
	host = 'https://learn.cytomine.be'
	public_key = getpass.getpass('Public key of cytomine account : ')   # u can get it on your account
	private_key = getpass.getpass('Private key of cytomine account : ') # u can get it on your account

	conn = Cytomine.connect(host, public_key, private_key, verbose=logging.ERROR)
	# Check https://docs.python.org/3.6/library/logging.html#logging-levels  to see other verbose level
	print(conn.current_user)

	# Get all images and annotations
	# Dict with img name and id
	imgs_dict_id = {}
	imgs_dict_height = {}
	projects = ProjectCollection().fetch()
	for project in projects:
		image_instances = ImageInstanceCollection().fetch_with_filter("project", project.id)
		for image in image_instances:
		    imgs_dict_id[image.filename] = image.id
		    imgs_dict_height[image.filename] = image.height
		    
	# Lines images
	p = "images_database"
	folders = [p+"/Example", p+"/eyes", p+"/NoEllipses", p+"/Team01", p+"/Team02", p+"/Team03", p+"/Team04", p+"/Team05",
		       p+"/Team06", p+"/Team07",p+"/Team08", p+"/Team09", p+"/Team10"]
	resp = input(f'The folders containing the images are "\n{folders}\n". Do you want to change them? [y, n] : ')
	if resp == "y":
		folders = []
		while True:
			path = input("What is the path of a folder containing the image? : ")
			folders.append(path)
			resp = input(f'The folder containing the images are "{folders}". Is there more? [y, n] : ')
			if resp == "n":
				break
				
	all_path = []
	all_names = []
	for i in range(len(folders)):
		all_names += os.listdir(folders[i])
		all_files_in_folder = os.listdir(folders[i])
		for fil in all_files_in_folder:
			all_path.append(folders[i]+"/"+fil)
	lines_img_names = sorted([img for img in all_names if (not "elps" in img) and img.endswith(".png")])
	lines_img_path = sorted([img for img in all_path if (not "elps" in img) and img.endswith(".png")])

	print("\nLoading the line images...")
	lines_img = []
	lines_name = []
	for name in lines_img_path:
		if 'pcb' in name:
			lines_img.append(cv2.imread(name, cv2.IMREAD_GRAYSCALE))
		else:
			lines_img.append(cv2.imread(name))    	
		lines_name.append(name.split("/")[-1])
		
	print("\nLoading the line annotations...")
	lines_ann = []
	for path in lines_img_path:
		name = path.split("/")[-1]
		annotations = AnnotationCollection()
		annotations.showWKT = True  # Ask to return WKT location (geometry) in the response
		annotations.showMeta = True  # Ask to return meta information (id, ...) in the response
		annotations.showGIS = True  # Ask to return GIS information (perimeter, area, ...) in the response
		if name in imgs_dict_id:
		    annotations.image = imgs_dict_id[name]
		else:
		    print(f"Image {name} not found.")
		annotations.fetch()
		img_ann = []
		err = False
		for annotation in annotations:
		    geometry = wkt.loads(annotation.location)
		    geometry_opencv = affine_transform(geometry, [1, 0, 0, -1, 0, imgs_dict_height[name]])
		    try:
		        pts = geometry_opencv.coords
		        img_ann.append([pts[0][0], pts[0][1], pts[1][0], pts[1][1]])
		    except Exception as e: # The lines were not labeled but the ellipses
		        err = True
		        print(f"Error with {name} : {e}.")
		if err == True:
		    for path in lines_img_path:
		    	if name in path:
		    		if 'pcb' in name:
		    			img_to_rem = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		    		else:
		    			img_to_rem = cv2.imread(path)
		    		for i in range(len(lines_img)):
		    			if np.array_equal(lines_img[i], img_to_rem):
		    				#last = i
		    				lines_img.pop(i)
		    				print(f"Error with {name} : removed from lines_img.")
		    				break
		    lines_name.remove(name)
		else:
		    lines_ann.append(img_ann)


	# Get value metrics
	def metricNaive(ar1, ar2):
		tmp = abs(ar1-ar2)
		return 1-np.mean(tmp)
		

	def metricFalseNeg(annoted, detected, error_false_neg = 0.2):
		num_to_get = np.sum(annoted)
		num_gotten = np.sum(np.multiply(annoted, detected))
		return min((num_gotten/num_to_get)/(1-error_false_neg), 1)
		

	def metricFalsePos(annoted, detected, error_false_positive = 0.20):
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
		m1 = metricFalseNeg(annoted, detected, error_false_neg)
		m2 = metricFalsePos(annoted, detected, error_false_positive)
		return (m1+m2)/2

	# Get image and seg detected
	print("\nStart evaluation...")
	mn = 0
	m1 = 0
	m2 = 0
	m3 = 0
	sys.stdout.write("\rProgress : 0% ")
	sys.stdout.flush()
	for i in range(len(lines_img)) :	
		sys.stdout.write(
			"\rProgress : %.2f%% " %((i+1) / len(lines_img) * 100))
		sys.stdout.flush()
		
		name = lines_name[i]
		type_img = None
		if "sudoku" in name:
			type_img = 'sudoku'
		elif 'pcb' in name:
			type_img = 'pcb'
		elif 'soccer' in name:
			type_img = 'soccer'
		elif 'road' in name or 'line_127710' in name or 'line_1277388' in name or 'line_image.00' in name:
			type_img = 'road'
		elif 'building' in name:
			type_img = 'building'
		else:
			print(f'Image unknown type : {name}')
		# detect segments
		img_edges, lines, img_edges_segment, img_segment = sd.segmentDetectorFinal(lines_img[i], type_img, 3)
		img_segment = img_segment[:,:,2]
		
		# get img annoted
		if type_img == 'pcb':
			img_annot =  cv2.cvtColor(lines_img[i], cv2.COLOR_GRAY2BGR)
		else :
			img_annot = lines_img[i].copy()
		img_annot =0*img_annot 
		for line in lines_ann[i]:
			cv2.line(img_annot, (int(line[0]), int(line[1])),(int(line[2]), int(line[3])), (0, 0, 255), 3)
		img_annot = img_annot[:,:,2]
		
		# compute metrics
		mn += metricNaive(img_annot/255, img_segment/255)
		m1_i = metricFalseNeg(img_annot/255, img_segment/255)
		m2_i = metricFalsePos(img_annot/255, img_segment/255)
		m1 += m1_i
		m2 += m2_i
		m3 += (m1_i+m2_i)/2
		
	mn = mn/len(lines_img)
	m1 = m1/len(lines_img)
	m2 = m2/len(lines_img)
	m3 = m3/len(lines_img)

	print(f"\n\nThe metrics are : \nmetricNaive = {mn}\nmetricFalseNeg = {m1}\nmetricFalsePos = {m2}\nmetricTot = {m3}")
	# metricNaive = 0.9449283841739091
	# metricFalseNeg = 0.7254580496023958
	# metricFalsePos = 0.19360395605349934
	# metricTot = 0.45953100282794757


if __name__ == '__main__':
	evaluate()
