import scipy

import cv2
import numpy as np
import glob
from PIL import Image as Imagee

import random
import matplotlib.pyplot as plt

from skimage.measure import compare_ssim

from tkinter import *
from tkinter.ttk import Progressbar
from tkinter import ttk

import scipy.signal as signal
import time
from scipy.spatial.distance import euclidean,cdist
from sklearn import cluster

# All song credit goes to Nintendo.
import winsound
winsound.PlaySound("bgmusic.wav", winsound.SND_ASYNC | winsound.SND_ALIAS | winsound.SND_LOOP )

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
        return False
    else:
        # skeletonPoints.delete(2)
        # objectPoints.delete(2)
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
        cv2.imshow("4: original video with outer contours drawn", image)

    outer_contours = np.array(outer_contours)
    outer_contours = np.reshape(outer_contours, (outer_contours.shape[0], 2))

    return outer_contours

def printOnScreen(originalVid, t ,score):
    fontColor = (255, 255, 255);
    fontScale = 1;
    lineType = 2;
    font = cv2.FONT_HERSHEY_SIMPLEX
    textPositionOriginal = (60, 100)

    timeText = "Time Left " + str(int(t))
    scoreText="Score "+str(score)
    timePosition = textPositionOriginal
    scorePosition=(60,150)
    cv2.putText(originalVid, timeText, timePosition, font, fontScale, fontColor, lineType)
    cv2.putText(originalVid, scoreText, scorePosition, font, fontScale, fontColor, lineType)


def getPeaks(processedImage, background, diffThreshHoldConst, originalImage, cutOffFreq, isVideoFrame=True, index = -1 , name="Camera", drawImage=True):
    pickedPeaks = None
    humanCenter = None
    (score, diffThreshold) = compare_ssim(processedImage, background, full=True)
    if (debugMode):
        cv2.imshow("2 Difference", diffThreshold)

    ############# Dilation + Erosion #############
    # Threshold
    if isVideoFrame:
        diffThreshold[diffThreshold < diffThreshHoldConst] = 0
        diffThreshold[diffThreshold > diffThreshHoldConst] = 1
        if (debugMode):
            cv2.imshow("2' Diff After Threshold", diffThreshold)
        # Erosion + Dilation. Erosion to fill holes with black
        dilationFilter = np.ones((5, 5), np.uint8)
        erosionVid = cv2.erode(diffThreshold, dilationFilter, iterations=4)
        dilationVid = cv2.dilate(erosionVid, dilationFilter, iterations=3)
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
            try:
                output_signal = np.transpose(signal.filtfilt(b, a, distance)).flatten()
            except:
                return None,None

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

                    if isVideoFrame:  #IF original video
                        # Head Body
                        cv2.line(originalImage, (int(pickedPeaks[3][0]), int(pickedPeaks[3][1])),
                                 (int(humanCenter[0][0]), int(humanCenter[0][1])), (255, 0, 0), 1)
                        # Left Body
                        cv2.line(originalImage, (int(pickedPeaks[1][0]), int(pickedPeaks[1][1])),
                                 (int(humanCenter[0][0]), int(humanCenter[0][1])), (255, 0, 0), 1)
                        # Right Body
                        cv2.line(originalImage, (int(pickedPeaks[0][0]), int(pickedPeaks[0][1])),
                                 (int(humanCenter[0][0]), int(humanCenter[0][1])), (255, 0, 0), 1)
                        # Leg Body
                        cv2.line(originalImage, (int(pickedPeaks[2][0]), int(pickedPeaks[2][1])),
                                 (int(humanCenter[0][0]), int(humanCenter[0][1])), (255, 0, 0), 1)
                    else:   # If showing frame
                        originalImage = cv2.resize(originalImage, (0,0), fx=0.25, fy=0.25)

                    if drawImage:
                        cv2.imshow("My Program"+str(isVideoFrame), np.fliplr(originalImage))

                    # if not isVideoFrame:
                    #     cv2.imwrite("Contoured "+str(index)+ ".png", originalImage)
                # plt.draw()
                # plt.pause(0.001)
                # plt.gcf().clear()
                # cv2.imshow("6 Skeleton All Points", image)
                # print("Object center is ", humanCenter)
    return pickedPeaks, humanCenter

def getFixedImagesPeaks(images, bgFilename, blurFactor, diffThreshHoldConst, cutOffFreq,drawImageinput=True):
    peaks = np.zeros((len(images),4,2))
    humanCenters = np.zeros((len(images), 2))
    background = np.array(cv2.imread(bgFilename))
    newSize = int(background.shape[0]/2)
    background = cv2.resize(background, (newSize,newSize))
    background = preprocessImage(background, blurFactor)

    for index, imgFilename in enumerate(images):
        image = np.array(cv2.imread(imgFilename))

        image = cv2.resize(image, (newSize,newSize))
        if debugMode:
            cv2.imshow("1 Original inside function", image)
        processed_img = preprocessImage(image, blurFactor)
        peaks[index], humanCenters[index] = getPeaks(processed_img, background, diffThreshHoldConst, image, cutOffFreq, False, index,drawImage=drawImageinput)

    return peaks,humanCenters

def preprocessImage(img, blurFactor):
    modified_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    modified_img = cv2.GaussianBlur(modified_img, (blurFactor, blurFactor), 0)
    return modified_img


def drawFunction(originalVid,peaks,humanCenter):
    return

    if peaks is None or humanCenter is None:
        return
    displayedScreen = np.zeros((originalVid.shape[0], originalVid.shape[1], 3), np.uint8)
    displayedScreen[:, 0:originalVid.shape[1]] = (255, 255, 255)

    # displayedScreen = originalVid.copy()

    #
    # # Right Hand
    # handSide = [[int(peaks[0][0]), int(peaks[0][1] - 20)], [int(peaks[0][0]), int(peaks[0][1] + 20)]]
    # centerSide = [[int(humanCenter[0][0]), int(humanCenter[0][1] - 20)],
    #               [int(humanCenter[0][0]), int(humanCenter[0][1] + 20)]]
    # contours = np.array([handSide[0], handSide[1], centerSide[1], centerSide[0]])  # COL Row
    # cv2.fillPoly(displayedScreen, pts=[contours], color=(255, 253, 209))
    #
    # # Left Hand
    # handSide = [[int(peaks[1][0]), int(peaks[1][1] - 20)], [int(peaks[1][0]), int(peaks[1][1] + 20)]]
    # contours = np.array([handSide[0], handSide[1], centerSide[1], centerSide[0]])  # COL Row
    # cv2.fillPoly(displayedScreen, pts=[contours], color=(255, 253, 209))
    #
    # # Legs
    # LegSide = [[int(peaks[2][0] - 30), int(peaks[2][1])], [int(peaks[2][0] + 30), int(peaks[2][1])]]
    # centerSideHorizontal = [[int(humanCenter[0][0] - 30), int(humanCenter[0][1])],
    #                         [int(humanCenter[0][0] + 30), int(humanCenter[0][1])]]
    # contours = np.array([LegSide[0], LegSide[1], centerSideHorizontal[1], centerSideHorizontal[0]])  # COL Row
    # cv2.fillPoly(displayedScreen, pts=[contours], color=(211, 218, 255))
    # cv2.line(displayedScreen, (int(humanCenter[0][0]), int(humanCenter[0][1] + 20)),
    #          (int(peaks[2][0]), int(peaks[2][1])), (0, 0, 0), 5)
    #
    # # Body
    # headSide = [[int(peaks[3][0]-25), int(peaks[3][1])], [int(peaks[3][0]+25), int(peaks[3][1])]]
    # contours = np.array([headSide[0], headSide[1], centerSideHorizontal[1], centerSideHorizontal[0]])  # COL Row
    # cv2.fillPoly(displayedScreen, pts=[contours], color=(255, 221, 229))
    #
    # # Head
    # FaceDiameter = 45
    # FaceHeight = 30
    # cv2.ellipse(displayedScreen, (int(peaks[3][0]), int(peaks[3][1])), (FaceDiameter, FaceHeight), 0, 0, 360, (255, 253, 209), -1)
    # cv2.circle(displayedScreen, (int(peaks[3][0]-(FaceDiameter/2)), int(peaks[3][1])), 5, (0, 0, 0), -1)
    # cv2.circle(displayedScreen, (int(peaks[3][0] + (FaceDiameter / 2)), int(peaks[3][1])), 5, (0, 0, 0), -1)
    # cv2.line(displayedScreen, (int(peaks[3][0]+ (FaceDiameter / 2)), int(peaks[3][1] + (FaceHeight/2))),
    #          (int(peaks[3][0]-(FaceDiameter / 2)), int(peaks[3][1]+(FaceHeight/2))), (0, 0, 0), 5)

    # Head
    cv2.circle(displayedScreen,(int(peaks[3][0]),int(peaks[3][1])), 5, (255,0,0), -1)
    # Body
    cv2.circle(displayedScreen, (int(humanCenter[0][0]), int(humanCenter[0][1])), 5, (255, 0, 0), -1)
    # Right
    cv2.circle(displayedScreen, (int(peaks[0][0]), int(peaks[0][1])), 5, (255, 0, 0), -1)
    # Left
    cv2.circle(displayedScreen, (int(peaks[1][0]), int(peaks[1][1])), 5, (255, 0, 0), -1)
    # Leg
    cv2.circle(displayedScreen, (int(peaks[2][0]), int(peaks[2][1])), 5, (255, 0, 0), -1)

    # Head Body
    cv2.line(displayedScreen,(int(peaks[3][0]),int(peaks[3][1])), (int(humanCenter[0][0]),int(humanCenter[0][1])),(255,0,0),5)
    # Left Body
    cv2.line(displayedScreen, (int(peaks[1][0]), int(peaks[1][1])), (int(humanCenter[0][0]), int(humanCenter[0][1])),(255,0,0),5)
    # Right Body
    cv2.line(displayedScreen, (int(peaks[0][0]), int(peaks[0][1])), (int(humanCenter[0][0]), int(humanCenter[0][1])),(255,0,0),5)
    # Leg Body
    cv2.line(displayedScreen, (int(peaks[2][0]), int(peaks[2][1])), (int(humanCenter[0][0]), int(humanCenter[0][1])),(255,0,0),5)


    #right left leg head
    cv2.imshow("DisplayedFigure", np.fliplr(displayedScreen))

def showStage(images, t=0, score=0, result=0):
    background = Imagee.open('stagebg.jpg')
    for index, imageName in enumerate(images):
        img = Imagee.open(imageName).convert("RGBA")
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2RGBA)
        img = Imagee.fromarray(img)
        background.paste(img, (60 + (240 * index), background.size[1] - img.size[1] - 70), img)
    # background.show()
    # background.close()

    background = np.array(background)
    if result == 1:  # Win
        background[:, :, 1] = 255
    elif result == 2:  # LOSE
        background[:, :, 2] = 255
    background = np.array(np.fliplr(background))
    printOnScreen(background, t , score)

    # background = cv2.resize(background,(0,0),fx=0.75, fy=0.75)  # TODO can comment it later
    cv2.imshow("bg stage", background)


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
    # eriosionIterations = 0 #5
    # dilationIterations = 0
    minRatioInImg = 500  #1771 #2234 #100
    BackgroundExtraction = 1    # 0=temporal Difference. 1=background Extraction
    global debugMode

    #################### plot ####################
    if debugMode:
        plt.ion()
        plt.show()

    fixedPoses = np.array(sorted(glob.glob("Pictures/*")))

    transparentPoses = np.array(sorted(glob.glob("Resized/*")))
    fixedPeaks, fixedCenters = getFixedImagesPeaks(fixedPoses, 'bg.jpg', blurFactor, diffThreshHoldConst, cutOffFreq,drawImageinput=False)

    prevFrame = originalVid.copy()
    prevGrayImg = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)
    prevGrayImg = cv2.GaussianBlur(prevGrayImg, (blurFactor, blurFactor), 0)
    # bckExtractor=cv2.createBackgroundSubtractorMOG2()   # Remove Later if not used

    ################################### Frame Processing Starts ###################################
    t0=time.time()
    gameType=0
    score = 0
    initalized = False
    cv2.destroyAllWindows()
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
            drawFunction(originalVid, peaks, humanCenter)
            # if compare(peaks, humanCenter,fixedPeaks[0], fixedCenters[0]):
            #     print("Matching!!")
            # else:
            #     print("Not matching. :(")
            if (debugMode):
                plt.draw()
                plt.pause(0.001)
                plt.gcf().clear()
                cv2.imshow("6 Skeleton All Points", originalVid)


        #################Game's logic#######################

        currentTime = time.time()
        timeDiff = currentTime - t0
        displayTime = 10
        hideTime = 2
        resultTime = 2

        if gameType == 0:  #Display All Mode

            if not initalized:  #Initialize
                chosenPoses = random.sample(range(len(fixedPoses)), 3)
                hiddenPose = random.sample(chosenPoses, 1)
                # For testing: hiddenPose=[imgIndex]

                currentCenter = fixedCenters[hiddenPose[0]]
                currentPeak = fixedPeaks[hiddenPose[0]]
                initalized = True
                t0 = time.time()
                matching = 0
            showStage(transparentPoses[chosenPoses] , displayTime-timeDiff , score) #TODO take timer
            if timeDiff > displayTime:
                cv2.destroyWindow('bg stage')
                gameType = 1
                t0 = time.time()
                print('game type is', gameType)

        elif gameType == 1:  # Hide All
            showStage([] , hideTime-timeDiff , score)  #TODO take timer
            if timeDiff > hideTime:
                cv2.destroyWindow('bg stage')
                gameType = 2
                t0 = time.time()
                print('game type is', gameType)
                initalized = False

        elif gameType == 2:  # Show 2 + Calculate score
            if not initalized:
                chosenPoses.remove(hiddenPose[0])
                initalized = True
                print([transparentPoses[hiddenPose[0]]])
                getFixedImagesPeaks([fixedPoses[hiddenPose[0]]],'bg.jpg', blurFactor, diffThreshHoldConst, cutOffFreq)
                # getPeaks(processedImage, background, diffThreshHoldConst, originalImage, cutOffFreq, isVideoFrame=True, index = -1)
            showStage(transparentPoses[chosenPoses], displayTime-timeDiff , score)  #TODO take timer, Write on screen, WRITE SCORE TOO
            if timeDiff > displayTime:
                cv2.destroyWindow('bg stage')
                gameType = 3
                t0 = time.time()
                print("game type is ", gameType)
                cv2.destroyWindow("My Program"+str(False))

            if peaks is not None and peaks.shape[0] >= 4:
                peaks[2] =[0,0]
                currentPeak[2] =[0,0]
                if compare(peaks, humanCenter, currentPeak, currentCenter):
                    print("Matching!!")
                    matching +=1
                    if matching >= 3:
                        score += 1
                        gameType=3
                        cv2.destroyWindow('bg stage')
                        t0 = time.time()
                        print("game type is ", gameType)
                        cv2.destroyWindow("My Program" + str(False))
                else:  #TODO remove those 2 lines
                    matching=0
                    print("Not Matching")

        elif gameType == 3:
            if matching>=3:
                showStage([transparentPoses[hiddenPose[0]]], score=score, result=1) #TODO write won
            else:
                showStage([transparentPoses[hiddenPose[0]]], score=score, result=2) #TODO write lose

            if timeDiff > resultTime:
                gameType = 0
                initalized = False
                t0 = time.time()
        ##############################################

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