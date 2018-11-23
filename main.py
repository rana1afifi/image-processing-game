import cv2
import numpy as np
import skimage.io as io
from scipy.signal import convolve2d

from skimage import morphology, color
import matplotlib.pyplot as plt

# For skeletonization
from skimage.morphology import skeletonize
from skimage import data
from skimage.util import invert

import imutils
import argparse
from skimage.measure import compare_ssim
from matplotlib.patches import Circle


from tkinter import *
from tkinter.ttk import Progressbar
from tkinter import ttk

import scipy.signal as signal


################################### Imports End ###################################




def startGameInterface():
    windowDimension = '350x300'

    ###### Create Window ######
    window = Tk()
    window.title("Welcome to LikeGeeks app")
    window.geometry(windowDimension)

    #### Assign Background ####

    # bg_image = PhotoImage(file = "coffee.jpeg")
    # bg_Label = Label (image=bg_image)
    # bg_Label.grid(row=0, column=0)

    # bg_label = Label(window, image=bg_image)
    # bg_label.image = bg_image
    # bg_label.place(x=0,y=0,relwidth=1,relheight=1)


    ###### Text Label ########
    # lbl = Label(window, text="Hello")
    # lbl.grid(column=0, row=0)

    ###### Button ############
    def clicked():
        print('hi')
        window.destroy()
        global openGame; openGame=1;
    btn = Button(window, text="Click Me",  bg="orange", fg="red", font=("Arial Bold", 10), command=clicked)  #, padx=5, pady=5
    btn.grid(column=0, row=0)

    ###### Progress Bar ######
    style = ttk.Style()
    style.theme_use('default')
    style.configure("black.Horizontal.TProgressbar", background='black')
    bar = Progressbar(window, length=200, style='black.Horizontal.TProgressbar')
    bar['value'] = 70
    bar.grid(column=1, row=1)

    window.mainloop()

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


def main():
    videoCam = cv2.VideoCapture(0)

    # Check web cam is working
    if videoCam.isOpened():
        ret, originalVid = videoCam.read()
    else:
        ret = False


    #################### Options ####################
    distanceSize = 200;
    cutOffFreq = 0.01
    diffThreshHoldConst =  0.8 #0.9
    blurFactor = 35 #15 #25
    eriosionIterations = 10 #5
    dilationIterations = 5
    minRatioInImg = 500  #1771 #2234 #100
    BackgroundExtraction = 1    # 0=temporal Difference. 1=background Extraction

    prevFrame = originalVid.copy()
    prevGrayImg = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)
    prevGrayImg = cv2.GaussianBlur(prevGrayImg, (blurFactor, blurFactor), 0)

    bckExtractor=cv2.createBackgroundSubtractorMOG2()

    #################### plot ####################
    plt.ion()
    plt.show()

    ################################### Frame Processing Starts ###################################
    while ret:
        ############# Store Previous Frame #############
        if BackgroundExtraction != 1:
            prevFrame = originalVid.copy()
            prevGrayImg = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)
            prevGrayImg = cv2.GaussianBlur(prevGrayImg, (blurFactor, blurFactor), 0)

        ############# Show Original Cam #############
        ret, originalVid = videoCam.read()
        cv2.imshow("1 Original", originalVid)

        ############# Gray #############
        grayImg = cv2.cvtColor(originalVid,cv2.COLOR_BGR2GRAY)
        blurredGrayImg = cv2.GaussianBlur(grayImg, (blurFactor, blurFactor), 0)

        ############# Difference #############
        (score, diff) = compare_ssim(blurredGrayImg, prevGrayImg, full=True)
        # diff=bckExtractor.apply(originalVid)

        # diff = (diff * 255).astype("uint8")
        cv2.imshow("2 Difference", diff)

        ############# Dilation + Erosion #############
        # Threshold
        diffThreshold = diff.copy()
        diffThreshold[diffThreshold<diffThreshHoldConst] = 0; diffThreshold[diffThreshold>diffThreshHoldConst] = 1;
        cv2.imshow("2' Diff After Thrshold", diffThreshold)
        # Erosion + Dilation. Erosion to fill holes with black
        dilationFilter = np.ones((5, 5), np.uint8)
        erosionVid = cv2.erode(diffThreshold, dilationFilter, iterations=eriosionIterations)
        dilationVid = cv2.dilate(erosionVid, dilationFilter, iterations=dilationIterations)
        cv2.imshow("3 After Erosion+Dilaition", dilationVid)
        # Show Canny Edge Detection         #TODO: change to border tracing
        edgeDetection2 = cv2.Canny(np.uint8(dilationVid*255), 30, 100)
        edgeDetection2 = invert(edgeDetection2)
        # cv2.imshow("4 Border", edgeDetection2)

        ############# Trace Borders - Like Contours #############
        # Get Indices of borders
        contourIndices = np.where(edgeDetection2<50)
        if BackgroundExtraction==1 or len(contourIndices[0]) > minRatioInImg: #If bgExtraction = 1, enter. OR if it is temportal diff, dont enter if lower than min ratio
            # Create Skeleton Window
            skeletonForm = edgeDetection2.copy()
            skeletonFormRGB = cv2.cvtColor(skeletonForm, cv2.COLOR_GRAY2RGB)

            # Get Human Center and show if there is someone in the image
            if len(contourIndices[0]) != 0:
                humanCenter = [int(np.average(contourIndices[0])),int(np.average(contourIndices[1]))]
                cv2.circle(skeletonFormRGB, (humanCenter[1], humanCenter[0]), 5, (0, 255, 0), -1)
                # cv2.line(skeletonFormRGB, (humanCenter[1], humanCenter[0]), (point2[0], point2[1]), (255, 0, 0), 2)  # Use this to Draw Line between two points
                # cv2.imshow("5 Skeleton", skeletonFormRGB)
                distance = np.sqrt(
                    np.power(humanCenter[0] - contourIndices[0], 2) + np.power(humanCenter[1] - contourIndices[1], 2))
                xAxis = np.linspace(0, len(distance) - 1, len(distance))

                # filter order , cutoff , type
                b, a = signal.butter(1, cutOffFreq, 'low')
                output_signal = signal.filtfilt(b, a, distance)

                #################### plot ####################
                if len(distance) != 0:
                    plt.plot(xAxis, output_signal)

                    ## Local Maximum Array of True & Falses
                    localMaximaTrueFalse = np.r_[True, output_signal[1:] > output_signal[:-1]] & np.r_[output_signal[:-1] > output_signal[1:], True]
                    ## Local Maxima Values Array
                    localMaximaValues = output_signal[np.where(localMaximaTrueFalse)]
                    ## Sorting and get Max 3
                    # localMaximaValues = localMaximaValues.argsort()#[-3:][::-1]
                    ## Get Indices of Local Maxima Values
                    LocalMaximaIndices = (np.array(np.where(localMaximaTrueFalse)))
                    LocalMaximaIndices = LocalMaximaIndices[0]  #LocalMaximaIndices[::]



                    for localMaximaIndex in LocalMaximaIndices:
                        plt.plot(xAxis[localMaximaIndex], output_signal[localMaximaIndex], marker = 'o')

                    chosenMaximaXIndeces = contourIndices[1][LocalMaximaIndices]
                    chosenMaximaYIndeces = contourIndices[0][LocalMaximaIndices]
                    averagePeakPointsX, averagePeakPointsY = averageMaximaPoints(chosenMaximaXIndeces, chosenMaximaYIndeces, distanceSize)

                    for iPoint in range(len(averagePeakPointsX)):
                        cv2.circle(skeletonFormRGB, (int(averagePeakPointsX[iPoint]), int(averagePeakPointsY[iPoint])), 5, (125, 125, 0), -1)

                    # plt.show()
                    plt.draw()
                    plt.pause(0.001)
                    plt.gcf().clear()
                    cv2.imshow("6 Skeleton All Points", skeletonFormRGB)

        if cv2.waitKey(1) == 27:
            break;

    cv2.destroyAllWindows()
    videoCam.release()




################################### Main ###################################

openGame = 0;

startGameInterface()
#openGame = 1;

if(openGame==1):
    if __name__=="__main__":
        main()



