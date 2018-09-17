import os
import cv2
import numpy as np




def write_to_file(file_name, content):

	fh = open(file_name, "a")
	fh.write(content)
	fh.close

	content = ""
	return content

# path = "../../data/Columbia_Gaze_Data_Set/"

path = ".."+ '\ '.strip() + ".."+ '\ '.strip() + "data"+ '\ '.strip() + "Columbia_Gaze_Data_Set"

folders = os.listdir(path)
# print folders

# img_list = []
# gaze_list = []
content = ""
file_name = "columbia_data.txt"

counter = 0
for folder in folders:
	if ".DS_Store" not in folder:
		files = os.listdir(path + folder)
		# print files
		for file in files:
			if ".jpg" in file:
				# print file
				parts = file.split("_")
				# print parts
				alpha = parts[3].split("V")[0]
				theta = parts[4].split("H")[0]
				# print theta
				# print alpha
				# img = cv2.imread(path + folder + "/" + file)
				img_path = os.path.abspath(path + folder + "/" + file)

				# print img.shape
				# img_list.append(img)
				# gaze_list.append([-float(theta), float(alpha)])
				line = img_path + " " + theta + " " + alpha + '\n'
				content += line
				counter += 1

				if counter % 100 == 0:
					content = write_to_file(file_name, content)

					print (counter)
	# outpath = os.path.join("./", "columbia_gaze_" + folder)
	# np.savez(outpath, img_list = img_list, gaze_list=gaze_list)
	# img_list = []
	# gaze_list = []
