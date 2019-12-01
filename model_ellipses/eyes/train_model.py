import cv2
import csv
import os
import numpy as np

database_directory = os.path.join(os.getcwd(), '../../images_database')
annotationFile = os.path.join(database_directory, 'CV2019_Annots.csv')

with open(annotationFile, mode='r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    result = []
    
    for row in csv_reader:
        print(row)
        print(row[0])
        if 'elps_eye' in row[0]:
            result.append(row[1:])

    # According to source code, Width >= Height. ret[1] = size Width and Heigth of rectangle, ret[0] = center of mass (height(vers le bas), width) ret[3]= angle in degrees [0 - 180[
    for eye in result:

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
        print(angle)

# black = cv2.imread("/home/pseudoless/Workspace/CVprojectPart2/images_database/Team01/" + pictureName, cv2.IMREAD_COLOR)
# cv2.imshow('Ellipse', black)
# cv2.ellipse(black, center, size, angle, 0, 360, (0, 0, 255), 1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print(result)
