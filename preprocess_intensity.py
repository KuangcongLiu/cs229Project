import numpy as np
import cv2
import os
import csv
from pathlib import Path
from datetime import datetime
from matplotlib import pyplot as plt
# % matplotlib inline

def preprocessForInOutNuclear(directory, id):
    blueName = directory +"/" + id + '_blue.png'
    greenName = directory +"/" + id + '_green.png'
    img1 = cv2.imread(blueName)
    img2 = cv2.imread(greenName)
    original_blue = img1
    original_green = img2
    
    # denoise the green graph 
    normalizedImg = np.zeros((img1.shape[0], img1.shape[1]))
    img2 = cv2.normalize(img2,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
    img2 = cv2.fastNlMeansDenoising(img2, None, 8, 7, 21)
    denoise_green = img2

    # make shape of the nuclear(blue image) clearer
    img1 = cv2.fastNlMeansDenoising(img1 , None, 6, 7, 21)
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((1,1),np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.9*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv2.watershed(img1, markers)
    img1[markers == -1] = [255,255,255]
    img1[markers == 1] = [255,255,255]

    greenAreaOfNuclearAndMembrane = np.sum(img2[markers == -1]) + np.sum(img2[markers == 1])
    greenAreaOutOfNuclearAndMembrane = np.sum(img2[markers == 2])
    areaOfNuclearAndMembrane = np.sum(markers == 1) + np.sum(markers == -1)

    if greenAreaOutOfNuclearAndMembrane == 0:
        intensity = 50
    else:
        intensity = greenAreaOfNuclearAndMembrane/greenAreaOutOfNuclearAndMembrane

    # fig = plt.figure(figsize=(8,8))

    # plt.subplot(241)
    # plt.title("Original Blue graph", fontsize=6)
    # plt.imshow(original_blue)

    # plt.subplot(242)
    # plt.title("Original Green graph", fontsize=6)
    # plt.imshow(original_green)

    # plt.subplot(243)
    # plt.title("Green graph after denoising", fontsize=6)
    # plt.imshow(denoise_green)

    # # img2[markers == -1] = [255,0,0]
    # # img2[markers == 1] = [0,255,0]

    # plt.subplot(243)
    # plt.title("Blue after segmentation", fontsize=6)
    # plt.imshow(img1, cmap='gray')

    # plt.subplot(244)
    # plt.title("Green with info about nuclear", fontsize=6)
    # plt.imshow(img2, cmap='gray')
    # plt.savefig("output1.png", dpi = 500)
    # plt.show()

    return intensity

def preprocessForAll(directory_in_str):

    directory = os.fsencode(directory_in_str)
    graph_dict = {}
    i = 0

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if "green" in filename:

            filenameList = filename.split('_')
            graph_id = filenameList[0]
            intensity = preprocessForInOutNuclear(directory_in_str, graph_id)
            graph_dict[graph_id] = intensity
            print(i, intensity)
            i += 1

    return graph_dict

def load_label(file_name):
    # label_dict = {'Id': list of labels}
    label_dict = {}
    firstline = True

    with open(file_name, mode='r') as infile:
        reader = csv.reader(infile)
        label_dict = {rows[0]:list(map(int, rows[1].split(' '))) for rows in reader if rows[0] != 'Id'}
    return label_dict


def separateTrainTo3Sets(label_dict, outputFileName1, outputFileName2, outputFileName3):
    myFile1 = open(outputFileName1, 'w')  
    with myFile1 as csv_file:  
        myFields = ['Id', 'Target']
        writer = csv.DictWriter(myFile1, fieldnames=myFields)    
        writer.writeheader()

        for j in label_dict:        
            # only inside nuclear 
            if set(label_dict[j]) & set(range(0,6)) and not (set(label_dict[j]) & set(range(6,28))):
                diction = {}
                diction['Id'] = j
                diction['Target'] = ' '.join(str(e) for e in label_dict[j])
                writer.writerow(diction)
    
    myFile2 = open(outputFileName2, 'w')  
    with myFile2 as csv_file:  
        myFields = ['Id', 'Target']
        writer = csv.DictWriter(myFile2, fieldnames=myFields)    
        writer.writeheader()

        for j in label_dict:       
            # only outside nuclear
            if not (set(label_dict[j]) & set(range(0,6))) and (set(label_dict[j]) & set(range(6,28))):
                diction = {}
                diction['Id'] = j
                diction['Target'] = ' '.join(str(e) for e in label_dict[j])
                writer.writerow(diction)
    
    myFile3 = open(outputFileName3, 'w')  
    with myFile3 as csv_file:  
        myFields = ['Id', 'Target']
        writer = csv.DictWriter(myFile3, fieldnames=myFields)    
        writer.writeheader()

        for j in label_dict:     
            # in and out of nuclear
            if set(label_dict[j]) & set(range(0,6)) and set(label_dict[j]) & set(range(6,28)):
                diction = {}
                diction['Id'] = j
                diction['Target'] = ' '.join(str(e) for e in label_dict[j])
                writer.writerow(diction)


def writeIntensityToCSV(outputFileName, graph_dict):
    myFile = open(outputFileName, 'w')  
    with myFile:  
        myFields = ['Id', 'Intensity']
        writer = csv.DictWriter(myFile, fieldnames=myFields)    
        writer.writeheader()

        for j in graph_dict: 
            diction = {}
            diction['Id'] = j
            diction['Intensity'] = graph_dict[j]
            writer.writerow(diction)

def separateTestTo3SetsDirectory(intensities_test, intensityThresholds):
    colors = ['red','green','blue','yellow']
    
    for j in intensities_test:        
        # only inside nuclear 
        if intensities_test[j] >= intensityThresholds[0]:
            directory = "testIn"        
        # only outside nuclear 
        elif intensities_test[j] <= intensityThresholds[1]:
            directory = "testOut"
        # In or Out
        else:
            directory = "testBoth"

        
        for color in colors:
            name_old = "../dataset/test/"+j+"_"+color+".png"
            name_new = "../dataset/"+directory+"/"+j+"_"+color+".png"
            os.rename(name_old, name_new)

def separateTestTo3SetsCSV(intensities_test, intensityThresholds, outputFileName1, outputFileName2, outputFileName3):
    
    myFile1 = open(outputFileName1, 'w')  
    with myFile1 as csv_file:  
        myFields = ['Id']
        writer = csv.DictWriter(myFile1, fieldnames=myFields)    
        writer.writeheader()

        for j in intensities_test:        
            # only inside nuclear 
            if intensities_test[j] >= intensityThresholds[0]:
                diction = {}
                diction['Id'] = j
                writer.writerow(diction)
    
    myFile2 = open(outputFileName2, 'w')  
    with myFile2 as csv_file:  
        myFields = ['Id']
        writer = csv.DictWriter(myFile2, fieldnames=myFields)    
        writer.writeheader()

        for j in intensities_test:       
            # only outside nuclear
            if intensities_test[j] <= intensityThresholds[1]:
                diction = {}
                diction['Id'] = j
                writer.writerow(diction)
    
    myFile3 = open(outputFileName3, 'w')  
    with myFile3 as csv_file:  
        myFields = ['Id']
        writer = csv.DictWriter(myFile3, fieldnames=myFields)    
        writer.writeheader()

        for j in intensities_test:     
            # in and out of nuclear
            if intensities_test[j] < intensityThresholds[0] and intensities_test[j] > intensityThresholds[1]:
                diction = {}
                diction['Id'] = j
                writer.writerow(diction)

if __name__ == "__main__":
    print(" Time start: "+ str(datetime.now()))
    pathString = "../../dataset/train"
    intensities_train = preprocessForAll(pathString)
    writeIntensityToCSV('../../output/intensities_train.csv', intensities_train)
    # for others to run
    label_dict_train = load_label('../../dataset/train.csv')
    # print(label_dict_train['00070df0-bbc3-11e8-b2bc-ac1f6b6435d0'])
    separateTrainTo3Sets(label_dict_train, "../../output/trainIn.csv", "../../output/trainOut.csv", "../../output/trainBoth.csv")

    pathString = "../../dataset/test"
    intensities_test = preprocessForAll(pathString)
    writeIntensityToCSV('../../output/intensities_test.csv', intensities_test)
    # separateTestTo3SetsCSV(intensities_test, intensityThresholds, "../output/testIn.csv", "../output/testOut.csv", "../output/testBoth.csv")
    print(" Time end: "+ str(datetime.now()))

    # for testing 
#     intensities_test={'a28d2bfc-bac6-11e8-b2b7-ac1f6b6435d0':10}
#     intensityThresholds=[3,1]
#     intensities_test = load_intensities('intensities_test.csv')
#     separateTestTo3SetsDirectory(intensities_test, intensityThresholds)
