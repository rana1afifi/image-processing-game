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
        comparison =  np.logical_and(skeletonAngle<=objectAngle+40, skeletonAngle>=objectAngle-40)
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
    cv2.imshow("original video with outer contours", image)

    outer_contours = np.array(outer_contours)
    outer_contours = np.reshape(outer_contours, (outer_contours.shape[0], 2))

    return outer_contours

def getPeaks(processed_img, background, diffThreshHoldConst, image, cutOffFreq):
    (score, diffThreshold) = compare_ssim(processed_img, background, full=True)
    if (debugMode):
        cv2.imshow("2 Difference", diffThreshold)

    ############# Dilation + Erosion #############
    # Threshold

    diffThreshold[diffThreshold < diffThreshHoldConst] = 255
    diffThreshold[diffThreshold != 255] = 0
    # diffThreshold=np.array(diffThreshold).astype('uint8')

    if (debugMode):
        cv2.imshow("2' Diff After Thrshold inside function", diffThreshold)

    diffThreshold = np.array(diffThreshold).astype('uint8')

    _, contours, _ = cv2.findContours(diffThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outer_contours = np.zeros((0, 0))
    if len(contours) > 0:

        outer_contours = getOuterContour(contours, image)

        if outer_contours.shape[0] > 1:
            humanCenter = np.average(outer_contours, axis=0).reshape((1, 2)).astype('int')
            cv2.circle(image, (humanCenter[0, 0], humanCenter[0, 1]), 5, (0, 255, 0), -1)
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



                # print("peaks = ", peaks)
                for iPoint in range(pickedPeaks.shape[0]):
                    cv2.circle(image, (int(pickedPeaks[iPoint, 0]), int(pickedPeaks[iPoint, 1])), 5, (0, 0, 255), -1)

                cv2.imshow("With Circle inside function ", image)

                # plt.draw()
                # plt.pause(0.001)
                # plt.gcf().clear()
                # cv2.imshow("6 Skeleton All Points", image)
                # print("Object center is ", humanCenter)
    return pickedPeaks

def getFixedImagesPeaks(images, bgFilename, blurFactor, diffThreshHoldConst, cutOffFreq):
    peaks = np.zeros((len(images),4,2))
    background = np.array(cv2.imread(bgFilename))
    background = cv2.resize(background, (int(background.shape[0] / 2), int(background.shape[1] / 2)))
    background = preprocessfixedImage(background, blurFactor)

    for index, imgFilename in enumerate(images):
        image = np.array(cv2.imread(imgFilename))

        image = cv2.resize(image, (int(image.shape[0] / 2), int(image.shape[1] / 2)))
        cv2.imshow("1 Original inside function", image)
        processed_img = preprocessfixedImage(image, blurFactor)
        peaks[index] = getPeaks(processed_img, background, diffThreshHoldConst, image, cutOffFreq)

    return peaks

def preprocessfixedImage(img, blurFactor):
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

    plt.ion()
    plt.show()
    getFixedImagesPeaks(['fig.jpg'], 'bg.jpg', blurFactor, diffThreshHoldConst, cutOffFreq)

    prevFrame = originalVid.copy()
    prevFrame = np.array(io.imread('bg.jpg'))
    prevFrame = cv2.resize(prevFrame, (int(prevFrame.shape[0] / 2), int(prevFrame.shape[1] / 2)))
    prevGrayImg = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)
    prevGrayImg = cv2.GaussianBlur(prevGrayImg, (blurFactor, blurFactor), 0)
    # bckExtractor=cv2.createBackgroundSubtractorMOG2()   # Remove Later if not used

    #################### plot ####################


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
        originalVid = cv2.imread('fig.jpg')
        originalVid = np.array(originalVid)
        originalVid = cv2.resize(originalVid, (int(originalVid.shape[0]/2), int(originalVid.shape[1]/2)))
        if(debugMode):
            cv2.imshow("1 Original", originalVid)

        ############# Gray #############
        grayImg = cv2.cvtColor(originalVid,cv2.COLOR_BGR2GRAY)
        blurredGrayImg = cv2.GaussianBlur(grayImg, (blurFactor, blurFactor), 0)

        ############# Difference #############
        (score, diff) = compare_ssim(blurredGrayImg, prevGrayImg, full=True)
        # diff=bckExtractor.apply(originalVid)
        if (debugMode):
            cv2.imshow("2 Difference", diff)

        ############# Dilation + Erosion #############
        # Threshold
        diffThreshold = diff.copy()
        diffThreshold[diffThreshold<diffThreshHoldConst] = 0; diffThreshold[diffThreshold>diffThreshHoldConst] = 1;
        if (debugMode):
            cv2.imshow("2' Diff After Thrshold", diffThreshold)
        # Erosion + Dilation. Erosion to fill holes with black
        dilationFilter = np.ones((5, 5), np.uint8)
        erosionVid = cv2.erode(diffThreshold, dilationFilter, iterations=eriosionIterations)
        dilationVid = cv2.dilate(erosionVid, dilationFilter, iterations=dilationIterations)
        cv2.imshow("3 After Erosion+Dilation", dilationVid)


        #### Contoursss
        # threshold = threshold_otsu(grayImg)
        # img=grayImg.copy()
        # img[img > threshold] = 0
        # img[img != 0] = 255
        #
        img=1-dilationVid
        img=np.array(img).astype('uint8')
        img[img==1]=255

        im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outer_contours=np.zeros((0,0))
        if len(contours)>0:
            contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
            outer_contours = max(contour_sizes, key=lambda x: x[0])[1]

            cv2.drawContours(originalVid, outer_contours, -1, (0, 255, 0), 3)
            cv2.imshow("original video with outer contours", originalVid)

            outer_contours = np.array(outer_contours)
            outer_contours = np.reshape(outer_contours, (outer_contours.shape[0], 2))
            # print(outer_contours)

        # Show Canny Edge Detection
        # edgeDetection2 = cv2.Canny(np.uint8(dilationVid*255), 30, 100)
        # edgeDetection2 = invert(edgeDetection2)
        # cv2.imshow("4 Border", edgeDetection2)

        ############# Trace Borders - Like Contours #############
        # Get Indices of borders
        # contourIndices = np.where(edgeDetection2<50)

        if BackgroundExtraction==1: ## or len(contourIndices[0]) > minRatioInImg: #If bgExtraction = 1, enter. OR if it is temportal diff, dont enter if lower than min ratio
            # Create Skeleton Window
            #skeletonForm = edgeDetection2.copy()
            #skeletonFormRGB = cv2.cvtColor(skeletonForm, cv2.COLOR_GRAY2RGB)

            # Get Human Center and show if there is someone in the image
            #  if len(contourIndices[0]) != 0:
            if outer_contours.shape[0] > 1:
                humanCenter = np.average(outer_contours,axis=0).reshape((1,2)).astype('int')
                cv2.circle(originalVid, (humanCenter[0,0], humanCenter[0,1]), 5, (0, 255, 0), -1)
                # cv2.line(skeletonFormRGB, (humanCenter[1], humanCenter[0]), (point2[0], point2[1]), (255, 0, 0), 2)  # Use this to Draw Line between two points
                # cv2.imshow("5 Skeleton", skeletonFormRGB)
                # distance = np.sqrt(np.power(humanCenter[0] - contourIndices[0], 2) + np.power(humanCenter[1] - contourIndices[1], 2))
                distance = cdist(humanCenter, outer_contours, metric='euclidean')

                xAxis = np.linspace(0, distance.shape[1] - 1, distance.shape[1])
                #plt.plot(xAxis,distance)

                # filter order , cutoff , type
                b, a = signal.butter(1, cutOffFreq, 'low')
                output_signal = np.transpose(signal.filtfilt(b, a, distance)).flatten()

                #################### plot & Draw New Points ####################
                if len(distance) != 0:
                #     if (debugMode):
                #         plt.plot(xAxis, output_signal)

                    ## Local Maximum Array of True & Falses
                    localMaximaTrueFalse = np.r_[True, output_signal[1:] > output_signal[:-1]] & np.r_[output_signal[:-1] > output_signal[1:], True]
                    ## Local Maxima Values Array
                    localMaximaValues = output_signal[np.where(localMaximaTrueFalse)]
                    ## Sorting and get Max 3
                    # localMaximaValues = localMaximaValues.argsort()#[-3:][::-1]
                    ## Get Indices of Local Maxima Values
                    LocalMaximaIndices = (np.array(np.where(localMaximaTrueFalse)))
                    LocalMaximaIndices = LocalMaximaIndices[0]  #LocalMaximaIndices[::]

                    # if (debugMode):
                    for localMaximaIndex in LocalMaximaIndices:
                        plt.plot(xAxis[localMaximaIndex], output_signal[localMaximaIndex], marker = 'o')

                    # Choose Center of Points through avg maxima indices
                    chosenMaximaXIndeces = outer_contours[LocalMaximaIndices][:,0]
                    chosenMaximaYIndeces = outer_contours[LocalMaximaIndices][:,1]

                    pickedPeaks = np.array(getKMeans(outer_contours[LocalMaximaIndices], k=4))
                    print("Current picked peaks \n" , pickedPeaks)
                    averagePeakPointsX, averagePeakPointsY = averageMaximaPoints(chosenMaximaXIndeces, chosenMaximaYIndeces, distanceSize)

                    for iPoint in range(pickedPeaks.shape[0]):
                        cv2.circle(originalVid, (int(pickedPeaks[iPoint,0]), int(pickedPeaks[iPoint,1])), 5, (0, 0, 255), -1)

                    # cv2.imshow("With Circle ",originalVid)
                    # for iPoint in len(averagePeakPointsX):
                    #     cv2.circle(originalVid, (int(averagePeakPointsX[iPoint]), int(averagePeakPointsY[iPoint])), 5,
                    #                (255, 255, 255), -1)
                    print("Object center is ", humanCenter)
                    if pickedPeaks.shape[0] >= 4:
                        rightHand = np.array(pickedPeaks[np.argmax(pickedPeaks[:, 0])])
                        pickedPeaks = np.delete(pickedPeaks, np.argmax(pickedPeaks[:, 0]), axis=0)
                        leftHand = np.array(pickedPeaks[np.argmin(pickedPeaks[:, 0])])
                        pickedPeaks = np.delete(pickedPeaks, np.argmin(pickedPeaks[:, 0]), axis=0)
                        leg = np.array(pickedPeaks[np.argmax(pickedPeaks[:, 1])])
                        head = np.delete(pickedPeaks, np.argmax(pickedPeaks[:, 1]), axis=0).flatten()
                        # print("current picked peaks ", pickedPeaks)
                        print(rightHand, leftHand, leg, head)

                        if compare(np.array([rightHand, leftHand, leg, head]).reshape((4,2)), humanCenter,np.array([[800,500],[112,161], [201,468],[316.5, 108]]), np.array([311, 249])):

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
