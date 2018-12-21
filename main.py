import scipy

import cv2
import numpy as np
import skimage.io as io
from scipy.signal import convolve2d
from skimage.filters import threshold_otsu

from skimage import morphology, color
import matplotlib.pyplot as plt

# For skeletonization
from skimage.morphology import skeletonize
from skimage import data
from skimage.util import invert

#import imutils
import argparse
from skimage.measure import compare_ssim
from matplotlib.patches import Circle


from tkinter import *
from tkinter.ttk import Progressbar
from tkinter import ttk

import scipy.signal as signal
import time
from scipy.spatial.distance import euclidean,cdist
from sklearn import cluster
################################### Imports End ###################################

#########################Function KMeans############################
def getKMeans(peaks, k=5):
    k = min(k, peaks.shape[0])
    kmeans = cluster.KMeans(n_clusters=k, random_state=0)
    kmeans.fit(peaks)
    return kmeans.cluster_centers_

######################### Function Compare ##############################
def compare(skeletonPoints,skeletonCenter,objectPoints,objectCenter):
    # print("skeleton shape is " , skeletonPoints.shape)
    # print("object shape is " , objectPoints.shape)
    if (skeletonPoints.shape[0]!=objectPoints.shape[0]):
        print("Different sizes")
        return  False
    else:
        skeletonDeltaXY = np.subtract(skeletonPoints, skeletonCenter)
        objectDeltaXY = np.subtract(objectPoints, objectCenter)
        skeletonAngle = np.rad2deg(np.arctan2(skeletonDeltaXY[:,1],skeletonDeltaXY[:,0]))
        objectAngle =  np.rad2deg(np.arctan2(objectDeltaXY[:,1],objectDeltaXY[:,0]))
        comparison =  np.logical_and(skeletonAngle<=objectAngle+20, skeletonAngle>=objectAngle-20)
        if(np.sum(comparison)==skeletonPoints.shape[0]):
            return True
        else:
            return False
    ### skeleton points and object points must be np.array with size x,2   while center must np.array with size 1,2 or list of 2 elements###

######################### Function Game Interface #######################
def startGameInterface():
    ###### Options ########
    gameTitle = "Welcome to Follow My Lead"
    backgroundImagePath = "photobg3.gif"
    backgroundImageColor = "black"
    windowAddedHeight = 55
    welcomeText = "Welcome To Follow My Lead"; welcomeTextBgColor="black";  welcomeTextColor = "white"
    buttonText = "Start Game"; buttonBgColor = "black"; buttonTextColor="white";
    progressBarWaitSeconds = 2;

    ###### Create Window ######
    window = Tk()
    window.resizable(False, False)
    window.title(gameTitle)

    ###### Assign Background Image #####
    photoBg = PhotoImage(file=backgroundImagePath)
    window.geometry(str(photoBg.width())+'x'+str(photoBg.height()+windowAddedHeight))
    window.configure(background=backgroundImageColor)
    Label(window, image=photoBg).grid(row=0, column=0, sticky=W)  # , bg="black"

    ###### Assign Welcome Text #####
    Label(window, text=welcomeText, bg=welcomeTextBgColor, fg=welcomeTextColor, font="none 12 bold").grid(row=1, column=0, sticky=W)

    ###### Button ############
    def clicked():
        oldTime = time.time(); finishTime = oldTime + progressBarWaitSeconds
        while (time.time() < finishTime):
            bar['value'] = 100 - int(100 *(finishTime-time.time())/progressBarWaitSeconds)
            window.update_idletasks()
        window.destroy()
        global openGame; openGame=1
    btn = Button(window, text=buttonText,  bg=buttonBgColor, fg=buttonTextColor, font=("Arial Bold", 10), command=clicked)  #, padx=5, pady=5
    btn.grid(column=0, row=0)

    ###### Progress Bar ######
    style = ttk.Style()
    style.theme_use('default')
    style.configure("black.Horizontal.TProgressbar", background='black')
    bar = Progressbar(window, length=400, style='black.Horizontal.TProgressbar')
    bar['value'] = 0
    bar.grid(column=0, row=2)

    #### Start Loop ######
    window.mainloop()

######################### Function Game Interface #######################
def makeSureBackGroundClearConfirm():

    ######## Options #########
    fontColor = (255, 255, 0); fontScale = 1;  lineType = 2; font = cv2.FONT_HERSHEY_SIMPLEX
    textPositionOriginal = (10, 100)
    welcomeText = 'First Make Sure To Leave The Screen/nThen Press Enter to Start/nOr Esc to Leave'
    betweenLineSpace = 100

    welcomeTextArray = welcomeText.split('/n')
    videoCam = cv2.VideoCapture(0)
    # Check web cam is working
    if videoCam.isOpened():
        ret, originalVid = videoCam.read()
    else:
        ret = False
    # Original Video
    while ret:
        ret, originalVid = videoCam.read()
        #### Write Text ####
        textPosition = textPositionOriginal
        for i in range(len(welcomeTextArray)):
            cv2.putText(originalVid, welcomeTextArray[i], textPosition, font, fontScale, fontColor, lineType)
            textPositionList = list(textPosition)     #Had to do this to change element in tuple
            textPositionList[1] += betweenLineSpace;
            textPosition = tuple(textPositionList)
        cv2.imshow("Original", originalVid)
        inputKey = cv2.waitKey(1)
        if inputKey == 13:
            global openGame; openGame=2
            break;
        elif inputKey == 27:
            break;
    cv2.destroyAllWindows()
    videoCam.release()

def getSkeletonParts (pickedPeaks):
    rightHand = np.array(pickedPeaks[np.argmax(pickedPeaks[:, 0])])
    pickedPeaks = np.delete(pickedPeaks, np.argmax(pickedPeaks[:, 0]), axis=0)
    leftHand = np.array(pickedPeaks[np.argmin(pickedPeaks[:, 0])])
    pickedPeaks = np.delete(pickedPeaks, np.argmin(pickedPeaks[:, 0]), axis=0)
    leg = np.array(pickedPeaks[np.argmax(pickedPeaks[:, 1])])
    head = np.delete(pickedPeaks, np.argmax(pickedPeaks[:, 1]), axis=0).flatten()
    return np.array([rightHand,leftHand,leg,head])

##### Function Calculate Distance and average nearby by points ##########
def averageMaximaPoints(chosenMaximaXIndeces,chosenMaximaYIndeces, distSize):
    resultArrayX = []
    resultArrayY = []

    while len(chosenMaximaXIndeces) != 0:
        firstPointX = chosenMaximaXIndeces[0]
        firstPointY = chosenMaximaYIndeces[0]
        chosenMaximaXIndeces = np.delete(chosenMaximaXIndeces, 0, None)
        chosenMaximaYIndeces = np.delete(chosenMaximaYIndeces, 0, None)

        distArray = np.sqrt( np.power((chosenMaximaXIndeces-firstPointX),2) + np.power((chosenMaximaYIndeces-firstPointY),2) )

        chosenIndices = np.where(distArray < distSize)

        if len(chosenIndices[0]) != 0:
            averageX = np.average(chosenMaximaXIndeces[chosenIndices])
            averageY = np.average(chosenMaximaYIndeces[chosenIndices])
            chosenMaximaXIndeces = np.delete(chosenMaximaXIndeces, chosenIndices, None)
            chosenMaximaYIndeces = np.delete(chosenMaximaYIndeces, chosenIndices, None)
        else:
            averageX = firstPointX
            averageY = firstPointY

        resultArrayX.append(averageX)
        resultArrayY.append(averageY)

    return resultArrayX , resultArrayY

def getOuterContour(contours,image):
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    outer_contours = max(contour_sizes, key=lambda x: x[0])[1]

    cv2.drawContours(image, outer_contours, -1, (0, 255, 0), 3)
    if debugMode:
        cv2.imshow("original video with outer contours", image)

    outer_contours = np.array(outer_contours)
    outer_contours = np.reshape(outer_contours, (outer_contours.shape[0], 2))

    return outer_contours

def getPeaks(processedImage, background, diffThreshHoldConst, originalImage, cutOffFreq, isVideoFrame=True):
    pickedPeaks = None
    humanCenter = None
    (score, diffThreshold) = compare_ssim(processedImage, background, full=True)
    if (debugMode):
        cv2.imshow("2 Difference", diffThreshold)

    ############# Dilation + Erosion #############
    # Threshold
    if isVideoFrame:
        diffThreshold[diffThreshold < diffThreshHoldConst] = 0;
        diffThreshold[diffThreshold > diffThreshHoldConst] = 1;
        if (debugMode):
            cv2.imshow("2' Diff After Threshold", diffThreshold)
        # Erosion + Dilation. Erosion to fill holes with black
        dilationFilter = np.ones((5, 5), np.uint8)
        erosionVid = cv2.erode(diffThreshold, dilationFilter, iterations=0)
        dilationVid = cv2.dilate(erosionVid, dilationFilter, iterations=0)
        if debugMode:
            cv2.imshow("3 After Erosion+Dilation", dilationVid)

        processedImage = np.array(1 - dilationVid).astype('uint8')
        processedImage[processedImage == 1] = 255
    else:
        diffThreshold[diffThreshold < diffThreshHoldConst] = 255
        diffThreshold[diffThreshold != 255] = 0
        processedImage = np.array(diffThreshold).astype('uint8')

    _, contours, _ = cv2.findContours(processedImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outer_contours = np.zeros((0, 0))
    if len(contours) > 0:

        outer_contours = getOuterContour(contours, originalImage)

        if outer_contours.shape[0] > 1:
            humanCenter = np.average(outer_contours, axis=0).reshape((1, 2)).astype('int')
            cv2.circle(originalImage, (humanCenter[0, 0], humanCenter[0, 1]), 5, (0, 255, 0), -1)
            distance = cdist(humanCenter, outer_contours, metric='euclidean')
            xAxis = np.linspace(0, (distance.shape[1]) - 1, distance.shape[1])
            # plt.plot(xAxis,distance)

            # filter order , cutoff , type
            b, a = signal.butter(1, cutOffFreq)
            output_signal = np.transpose(signal.filtfilt(b, a, distance)).flatten()

            #################### plot & Draw New Points ####################
            if len(distance) != 0:
                if (debugMode):
                    plt.plot(xAxis, output_signal)

                ## Local Maximum Array of True & Falses
                localMaximaTrueFalse = np.r_[True, output_signal[1:] > output_signal[:-1]] & np.r_[
                    output_signal[:-1] > output_signal[1:], True]
                ## Get Indices of Local Maxima Values
                LocalMaximaIndices = np.array(np.where(localMaximaTrueFalse))[0]

                if (debugMode):
                    for localMaximaIndex in LocalMaximaIndices:
                        plt.plot(xAxis[localMaximaIndex], output_signal[localMaximaIndex], marker='o')

                pickedPeaks = np.array(getKMeans(outer_contours[LocalMaximaIndices], k=4))
                if pickedPeaks.shape[0] >= 4:
                    pickedPeaks = getSkeletonParts(pickedPeaks)
                    # print("peaks = ", peaks)
                    for iPoint in range(pickedPeaks.shape[0]):
                        cv2.circle(originalImage, (int(pickedPeaks[iPoint, 0]), int(pickedPeaks[iPoint, 1])), 5, (0, 0, 255), -1)

                    cv2.imshow("With Circle "+ ' for '+ str(isVideoFrame), np.fliplr(originalImage))

                # plt.draw()
                # plt.pause(0.001)
                # plt.gcf().clear()
                # cv2.imshow("6 Skeleton All Points", image)
                # print("Object center is ", humanCenter)
    return pickedPeaks, humanCenter

def getFixedImagesPeaks(images, bgFilename, blurFactor, diffThreshHoldConst, cutOffFreq):
    peaks = np.zeros((len(images),4,2))
    humanCenters = np.zeros((len(images), 2))
    background = np.array(cv2.imread(bgFilename))
    background = cv2.resize(background, (int(background.shape[0] / 2), int(background.shape[1] / 2)))
    background = preprocessImage(background, blurFactor)

    for index, imgFilename in enumerate(images):
        image = np.array(cv2.imread(imgFilename))

        image = cv2.resize(image, (int(image.shape[0] / 2), int(image.shape[1] / 2)))
        if debugMode:
            cv2.imshow("1 Original inside function", image)
        processed_img = preprocessImage(image, blurFactor)
        peaks[index], humanCenters[index] = getPeaks(processed_img, background, diffThreshHoldConst, image, cutOffFreq, False)

    return peaks,humanCenters

def preprocessImage(img, blurFactor):
    modified_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    modified_img = cv2.GaussianBlur(modified_img, (blurFactor, blurFactor), 0)
    return modified_img

######################### Function Main Game #######################
def main():
    videoCam = cv2.VideoCapture(0)
    originalVid = []
    # Check web cam is working
    if videoCam.isOpened():
        ret, originalVid = videoCam.read()
    else:
        ret = False

    #################### Options ####################
    distanceSize = 200;
    cutOffFreq = 0.1
    diffThreshHoldConst =  0.8 #0.9
    blurFactor = 35 #15 #25
    eriosionIterations = 0 #5
    dilationIterations = 0
    minRatioInImg = 500  #1771 #2234 #100
    BackgroundExtraction = 1    # 0=temporal Difference. 1=background Extraction
    global debugMode

    #################### plot ####################
    if debugMode:
        plt.ion()
        plt.show()

    fixedPeaks, fixedCenters = getFixedImagesPeaks(['fig.jpg'], 'bg.jpg', blurFactor, diffThreshHoldConst, cutOffFreq)

    prevFrame = originalVid.copy()
    prevGrayImg = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)
    prevGrayImg = cv2.GaussianBlur(prevGrayImg, (blurFactor, blurFactor), 0)
    # bckExtractor=cv2.createBackgroundSubtractorMOG2()   # Remove Later if not used

    ################################### Frame Processing Starts ###################################
    while ret:
        ############# Store Previous Frame #############
        if BackgroundExtraction != 1:
            prevFrame = originalVid.copy()
            prevGrayImg = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)
            prevGrayImg = cv2.GaussianBlur(prevGrayImg, (blurFactor, blurFactor), 0)

        ############# Show Original Cam #############
        ret, originalVid = videoCam.read()
#       ret = True
        if(debugMode):
            cv2.imshow("1 Original", originalVid)

        ############# Gray #############
        grayImg = cv2.cvtColor(originalVid,cv2.COLOR_BGR2GRAY)
        blurredGrayImg = cv2.GaussianBlur(grayImg, (blurFactor, blurFactor), 0)
        peaks, humanCenter = getPeaks(blurredGrayImg, prevGrayImg, diffThreshHoldConst, originalVid, cutOffFreq, isVideoFrame=True)
        if peaks is not None and peaks.shape[0] >= 4:
            if compare(peaks, humanCenter,fixedPeaks[0], fixedCenters[0]):
                print("Matching!!")
            else:
                print("Not matching. :(")
            if (debugMode):
                plt.draw()
                plt.pause(0.001)
                plt.gcf().clear()
                cv2.imshow("6 Skeleton All Points", originalVid)

        if cv2.waitKey(1) == 27:
            break;

    cv2.destroyAllWindows()
    videoCam.release()






################################### Main ###################################

debugMode = False;
openGame = 0;   # If you wanna skip interface, assign this to 2

startGameInterface()

if(openGame==1):
    makeSureBackGroundClearConfirm()

if(openGame==2):
    if __name__=="__main__":
        main()