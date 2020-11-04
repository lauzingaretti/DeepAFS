"""
Function to load and segment image using a thresholding method
"""

# requiriments

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
import pytesseract
from pytesseract import image_to_string
import imutils
import glob



class load_image(object):
    """
    This class read image data into memory.
    """
    def __init__(self, fileName,Gray=False):
        self.fileName = fileName
        self.Gray =Gray
        self.img = cv2.imread(fileName)
        if Gray is True:
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.shape=self.img.shape



    def __iter__(self):
        return self

    def histo(self):
        '''
        function to plot image histogram

        '''
        im = self.img
        if (im.shape[2]==3):
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        hist,bins = np.histogram(im.ravel(),256,[0,256])
        plt.hist(im.ravel(),256,[0,256]); plt.show()



    def agis(self, kernelsize=3, threshold="Adap", ud=10 ,dilate=True,erode=True) :
        '''
        file name: path to img
        histogram: can you see the img hist?
        kernelsize: to medianBlur filtering (Default=3)
        threshold: method to img thresholding OTSU,BIN, Adapt,  Gaussian, i.e. authomatic, manual, automatic use of hist info, adaptative, gaussian
        ud: only works if threshold==BIN, a user defined number between  0 255 to define  the threshold value
        dilate: True or False
        erode: True or False
        '''
        threshold = threshold.lower()
        if threshold in ["adap","bin","otsu","gaussian"] is False:
            threshold="adap"
            print("warning, threshold should be one of adap,adap2,bin,otsu,gaussian")
        im = self.img
        if (im.shape[2]==3):
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        if kernelsize>0:
            blur = cv2.medianBlur(im,kernelsize,0)
        if kernelsize==0:
            blur = im
        if threshold=="otsu":
            x=(np.mean(im))
            x= x.astype(int)
            if x < 255/2:
                ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            else:
                ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

            kernel = np.ones((16,16), np.uint8)

            th3 = cv2.morphologyEx(th3, cv2.MORPH_OPEN,kernel)
            if dilate is True:
                th3 = cv2.dilate(th3, None, iterations=4)
            if erode is True:
                th3 = cv2.erode(th3, None, iterations=8)

        if threshold=="bin":
            print("remember to select proper lower limit for binary thresholding, the histogram information is helpful ")
            x=(np.mean(im))
            x= x.astype(int)
            if x < 255/2:
                th3= cv2.threshold(blur,ud,255,cv2.THRESH_BINARY)[1]
            else:
                th3= cv2.threshold(blur,ud,255,cv2.THRESH_BINARY_INV)[1]

            ret3=ud

            kernel = np.ones((16,16), np.uint8)

            th3 = cv2.morphologyEx(th3, cv2.MORPH_OPEN,kernel)

            if dilate is True:
                th3 = cv2.dilate(th3, None, iterations=4)
            if erode is True:
                th3 = cv2.erode(th3, None, iterations=8)

        if threshold=="adap":
            x=(np.mean(im))
            x= x.astype(int)
            if x < 255/2:
                ret3,th3= cv2.threshold(blur,x,255,cv2.THRESH_BINARY)
            else:
                ret3,th3= cv2.threshold(blur,x,255,cv2.THRESH_BINARY_INV)
            kernel = np.ones((16,16), np.uint8)

            th3 = cv2.morphologyEx(th3, cv2.MORPH_OPEN,kernel)

            #ret3=x
            if dilate is True:
                th3 = cv2.dilate(th3, None, iterations=4)
            if erode is True:
                th3 = cv2.erode(th3, None, iterations=8)
        if threshold=="gaussian":
            x=(np.mean(im))
            x= x.astype(int)
            if x < 255/2:
                th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,2)
            else:
                th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,15,2)
            kernel = np.ones((16,16), np.uint8)

            th3 = cv2.morphologyEx(th3, cv2.MORPH_OPEN,kernel)
            ret3=0
            if dilate is True:
                th3 = cv2.dilate(th3, None, iterations=4)
            if erode is True:
                th3 = cv2.erode(th3, None, iterations=8)




        self.thr=ret3
        self.mask=th3
        return(self)



    def acis(self, criteria="k-means", k=3,n_iter=10) :
        '''
        file name: path to img
        criteria: k-means (This is the only criteria right now)
        k: number of clusters
        n_iter number of iterations
        '''
        im = self.img
        if (im.shape[2]==1):
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        pixel_values = im.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        centroid_initialization_strategy = cv2.KMEANS_RANDOM_CENTERS

        _, labels, centers = cv2.kmeans(pixel_values,
                                    k,
                                    None,
                                    stop_criteria,
                                    n_iter,
                                    centroid_initialization_strategy)

        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]

        # Debemos reestructurar el arreglo de datos segmentados con las dimensiones de la imagen original.
        segmented_image = segmented_data.reshape(im.shape)

        mask_1=[]
        for cluster in np.unique(labels):
            mask= np.zeros((im.shape[0],im.shape[1]),dtype="uint8")
            La=labels.reshape(im.shape[0],im.shape[1])
            mask[La== cluster] = 255
            mask_1.append(mask)
        self.km=segmented_image
        self.mask=mask_1
        return(self)
