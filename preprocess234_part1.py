import numpy as np
import cv2
import os
import csv
from pathlib import Path
from datetime import datetime
from matplotlib import pyplot as plt
from skimage import measure
import pandas as pd
# % matplotlib inline

def preprocessForFeature234(directory, id):
    
    blueName = directory +"/" + id + '_blue.png'
    greenName = directory +"/" + id + '_green.png'
    yellowName = directory +"/" + id + '_yellow.png'
    redName = directory +"/" + id + '_red.png'
    
    img_blue = cv2.imread(blueName)
    img_green = cv2.imread(greenName)
    img_yellow = cv2.imread(yellowName)
    img_red = cv2.imread(redName)
    
    original_blue = img_blue
    original_green = img_green
    original_yellow = img_yellow
    original_red = img_red
    
    # denoise the green, yellow, red graph, similar to original
    normalizedImg = np.zeros((img_green.shape[0], img_green.shape[1]))
    img_green = cv2.normalize(img_green,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
    img_green = cv2.fastNlMeansDenoising(img_green, None, 8, 7, 21)
    denoise_green = cv2.cvtColor(img_green, cv2.COLOR_BGR2GRAY)
    
    normalizedImg = np.zeros((img_yellow.shape[0], img_yellow.shape[1]))
    img_yellow = cv2.normalize(img_yellow,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
    img_yellow = cv2.fastNlMeansDenoising(img_yellow, None, 8, 7, 21)
    denoise_yellow = cv2.cvtColor(img_yellow, cv2.COLOR_BGR2GRAY)
    
    normalizedImg = np.zeros((img_red.shape[0], img_red.shape[1]))
    img_red = cv2.normalize(img_red,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
    img_red = cv2.fastNlMeansDenoising(img_red, None, 8, 7, 21)
    denoise_red = cv2.cvtColor(img_red, cv2.COLOR_BGR2GRAY)

    normalizedImg = np.zeros((img_blue.shape[0], img_blue.shape[1]))
    img_blue = cv2.normalize(img_blue,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
    img_blue = cv2.fastNlMeansDenoising(img_blue, None, 8, 7, 21)
    denoise_blue = cv2.cvtColor(img_blue, cv2.COLOR_BGR2GRAY)   
    
    feature2 = measure.compare_ssim(denoise_green, denoise_red)
    feature3 = measure.compare_ssim(denoise_green, denoise_yellow)
    feature4 = measure.compare_ssim(denoise_green, denoise_blue)
    feature5 = np.sum(denoise_green)*1.0 / np.sum(denoise_yellow) 
    background = np.average(denoise_green)*0.05
    feature6 = np.sum(denoise_green > background)
    
    return [feature2, feature3, feature4, feature5, feature6]

def preprocessForAll234(filename, directory_in_str, startIdx, endIdx):

    df = pd.read_csv(filename)
    df.head()
    IDs = df.Id
    IDs = IDs[startIdx: endIdx+1]

    graph_dict = {}
    i = 0

    for Id in IDs:
        features = preprocessForFeature234(directory_in_str, Id)
        graph_dict[Id] = features
        print(i, " Graph ", Id, "Features234: ", features, " Time now: "+ str(datetime.now()))
        i += 1

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

if __name__ == "__main__":
    print(" Time start: "+ str(datetime.now()))
    headerName = ['Id', 'SSIMGreenRed', 'SSIMGreenYellow', 'SSIMGreenBlue', 'IntensityGreenOverYellow', 'NumberOfGreen']
    pathString = "../../dataset/train"
    train_features234 = preprocessForAll234('../../dataset/train.csv', pathString, 0, 10358)
    writeToCSV('../../output/train_features234_part1.csv', headerName, train_features234)
    print(" Time end: "+ str(datetime.now()))


