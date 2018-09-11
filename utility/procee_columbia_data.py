import os
import cv2
import numpy as np

path = "../../data/Columbia_Gaze_Data_Set/"
folders = os.listdir(path)
# print folders

img_list = []
gaze_list = []
counter = 0
for folder in folders:
    files = os.listdir(path + folder)
    # print files
    for file in files:
        if ".jpg" in file:
            counter += 1
            if counter % 50 == 0:
                print counter

            # print file
            parts = file.split("_")
            # print parts
            alpha = parts[3].split("V")[0]
            theta = parts[4].split("H")[0]
            # print theta
            # print alpha
            img = cv2.imread(path + folder + "/" + file)
            # print img.shape
            img_list.append(img)
            gaze_list.append([-float(theta), float(alpha)])

    outpath = os.path.join("./", "columbia_gaze_" + folder)
    np.savez(outpath, img_list = img_list, gaze_list=gaze_list)
    img_list = []
    gaze_list = []
