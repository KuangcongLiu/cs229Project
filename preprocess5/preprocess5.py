import numpy as np
import cv2
import os
import csv
from pathlib import Path
from datetime import datetime
from matplotlib import pyplot as plt
import pandas as pd

# % matplotlib inline

def segmentBlue(directory, id, plot = False):
    
    blueName = directory +"/" + id + '_blue.png'
    img2 = cv2.imread(blueName)

    if plot:
        plt.subplot(241),plt.imshow(img2)
    
    # denoise the green graph 
    normalizedImg = np.zeros((img2.shape[0], img2.shape[1]))
    img = cv2.normalize(img2,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
    img = cv2.GaussianBlur(img,(3,3),0)
    img = cv2.fastNlMeansDenoising(img, None, 8, 7, 21)

    img = cv2.GaussianBlur(img,(5,5),0)
    img = cv2.fastNlMeansDenoising(img,None,6,7,21)
    
    img = 2 * img

    if plot:
        plt.subplot(242),plt.imshow(img, cmap='gray')


    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


    # noise removal
    kernel = np.ones((1,1),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # plt.subplot(248),plt.imshow(opening, cmap='gray')
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]
    img[markers == 1] = [0,255,0]

    # print(markers)
    img2[markers > 1] = [0,0,0]
    img2[markers == -1] = [255,255,255]
    img2[markers == 1] = [255,255,255]


    reverse = 255-img2
    gray = cv2.cvtColor(reverse,cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 200, 255, 0)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    contours_area = []
    # calculate area and filter into new array
    for con in contours:
        area = cv2.contourArea(con)
        if 150 < area < 100000:
            contours_area.append(con)

    return contours_area



def preprocessForFeature5(directory, id, plot = False):
    
    greenName = directory +"/" + id + '_green.png'
    green_image = cv2.imread(greenName)
    img2 = cv2.imread(greenName)

    if plot:
        plt.subplot(241),plt.imshow(img2)
    
    # denoise the green graph 
    normalizedImg = np.zeros((img2.shape[0], img2.shape[1]))
    img = cv2.normalize(img2,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
    img = cv2.GaussianBlur(img,(3,3),0)
    img = cv2.fastNlMeansDenoising(img, None, 8, 7, 21)

    img = cv2.GaussianBlur(img,(5,5),0)
    img = cv2.fastNlMeansDenoising(img,None,6,7,21)
    
    img = 2 * img

    if plot:
        plt.subplot(242),plt.imshow(img, cmap='gray')


    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


    # noise removal
    kernel = np.ones((1,1),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # plt.subplot(248),plt.imshow(opening, cmap='gray')
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]
    img[markers == 1] = [0,255,0]

    # print(markers)
    img2[markers > 1] = [0,0,0]
    img2[markers == -1] = [255,255,255]
    img2[markers == 1] = [255,255,255]

    reverse = 255-img2
    gray = cv2.cvtColor(reverse,cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 200, 255, 0)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    green_contours = []
    # calculate area and filter into new array
    for con in contours:
        area = cv2.contourArea(con)
        if 150 < area < 100000:
            green_contours.append(con)

    sum_compactness = 0
    sum_area = 0
    sum_eccentricity = 0
    sum_variance = 0
    for con in green_contours:
        perimeter = cv2.arcLength(con, True)
        area = cv2.contourArea(con)
        sum_area += area
        sum_compactness += (perimeter)**2 / (4*np.pi*area)

        (x,y),(ma,MA),angle = cv2.fitEllipse(con)
        # print(ma)
        # print(MA)
        sum_eccentricity += ma / MA

        x,y,w,h = cv2.boundingRect(con)
        matrix = green_image[x:x+w, y:y+h]
        sum_variance += np.var(matrix[matrix!=0])


    cv2.drawContours(green_image, green_contours, -1, (0,255,0), 3)


    if len(green_contours) != 0:
        average_area_green = sum_area / len(green_contours)
        average_compactness = sum_compactness / len(green_contours)
        average_eccentricity = sum_eccentricity / len(green_contours)
        average_variance = sum_variance / len(green_contours)
    else:
        average_area_green, average_compactness, average_eccentricity, average_variance = 0, 0, 0, 0

    
    blueName = directory +"/" + id + '_blue.png'
    blue_contours = segmentBlue(directory, id, plot = False)
    blue_image = cv2.imread(blueName)
    cv2.drawContours(blue_image, blue_contours, -1, (0,255,0), 3)


    # feature5 = len(green_contours)/ len(blue_contours)
    sum_area = 0
    for con in blue_contours:
        sum_area += cv2.contourArea(con)

    if len(blue_contours) != 0:
        average_area_blue = sum_area/len(blue_contours)
        feature5 = average_area_green / average_area_blue
    else:
        feature5 = 0

    sum_mindist = 0

    if len(blue_contours) == 0:
        average_dist = 0
    else:
        for gc in green_contours:
            dist = []
            x,y,w,h = cv2.boundingRect(gc)
            gx = x + w/2
            gy = y + h/2
            for bc in blue_contours:
                x,y,w,h = cv2.boundingRect(bc)
                bx = x + w/2
                by = y + h/2
                dist.append((gx-bx)**2 + (gy-by)**2)
            sum_mindist += np.sqrt(min(dist))

        if len(green_contours) != 0:
            average_dist = sum_mindist / len(green_contours)
        else:
            average_dist = 0

    
# 
    # print(feature5)
    # print(average_compactness)
    # print(average_eccentricity)
    # print(average_dist)
    # print(average_variance)

    if plot:

        plt.subplot(243),plt.imshow(img2, cmap='gray')
        plt.subplot(244),plt.imshow(green_image, cmap='gray')
        plt.subplot(245),plt.imshow(blue_image, cmap='gray')

        plt.show()


    
    return [feature5, average_compactness, average_eccentricity, average_dist, average_variance]



def preprocessForAll5(filename, directory_in_str, startIdx, endIdx):

    df = pd.read_csv(filename)
    df.head()
    IDs = df.Id
    IDs = IDs[startIdx: endIdx+1]


    graph_dict = {}

    for Id in IDs:
        features = preprocessForFeature5(directory_in_str, Id)
        graph_dict[Id] = features
        print(Id, features, " Time now: "+ str(datetime.now()))

    directory = os.fsencode(directory_in_str)
    
    return graph_dict

def writeToCSV(outputFileName, headerName, graph_dict):
    myFile = open(outputFileName, 'w')  
    with myFile:  
        myFields = headerName 
        writer = csv.DictWriter(myFile, fieldnames=myFields)    
        writer.writeheader()

        for j in graph_dict: 
            diction = {}
            diction[headerName[0]] = j
            for i in range(1, len(headerName)): 
                diction[headerName[i]] = graph_dict[j][i-1]
                
            writer.writerow(diction)
