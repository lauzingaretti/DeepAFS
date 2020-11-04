# requiriments
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pytesseract
from pytesseract import image_to_string
import imutils
import utils
from PIL import Image, ImageDraw, ImageFilter
from .load_image import load_image
from .scanIm import imgObjects
from sklearn.cluster import KMeans
import pandas as pd
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries

def Linear_Descriptors(self, both_sizes=True,leaves=True, mask_=False):
    """
    Function to compute the linear descriptors from img
    self an object the class load_image
    both_sizes if true it detects both sizes objects (note that it is optimized to
    fruit vegetable classification and may not work properly for other purposes)
    leaves (optimized for strawberries) Do the fruits have the calix? (only remove leaves for the outside fruits)
    Note leaves can be used to perform segmentation inside the fruit pattern, for instance in apple.
    mask_: logical if is true, the slic use the mask to segment the image.
    """
    Objects =self.objects
    shape= self.shape
    mascaras=self.all_masks
    f=self.id
    height=shape[0]
    width=shape[1]
    seg=[]
    imgcl=[]

    if both_sizes is True:
        data=pd.DataFrame()
        for Example in Objects:
            Example=cv2.cvtColor(Example,cv2.COLOR_BGR2RGB)

            imgray = cv2.cvtColor(Example, cv2.COLOR_RGB2GRAY)

            LAB=cv2.cvtColor(Example, cv2.COLOR_RGB2LAB)

            l_channel,a_channel,b_channel = cv2.split(LAB)
            a=np.mean(imgray[np.nonzero(imgray)])
            a1=np.std(imgray[np.nonzero(imgray)])
            temp = pd.DataFrame({
                    'Mean': a,
                    'sd': a1,
                    'ratio': a1/a},index=[0])
            data= pd.concat([data, temp])
        kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)
        out=kmeans.fit(data)
        label=out.labels_
        data['label']=label
        sum=data.groupby('label').mean()
        ####leaves not

        if leaves is True:
            i=0
            for img in Objects:
                #idxmin
                if (data.iloc[i,3]==sum['Mean'].idxmin()):
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                    LAB= cv2.cvtColor(img,cv2.COLOR_RGB2LAB)
                    if mask_ is False:
                        if (data.iloc[i,3]>=0.7):
                            segments = slic(img, n_segments=10, sigma=3, start_label=1,compactness=20)
                        else:
                            segments = slic(img, n_segments=5, sigma=3, start_label=1,compactness=20)
                    else:
                        if (data.iloc[i,3]>=0.7):
                            segments = slic(img, n_segments=3, sigma=3, start_label=1,compactness=20, mask=mascaras[i])
                        else:
                            segments = slic(img, n_segments=3, sigma=3, start_label=1,compactness=20,mask=mascaras[i])

                        #20
                        #mask=mascaras[i]

                    for label in np.unique(segments):
                        mask = np.zeros((img.shape[0],img.shape[1]), dtype="uint8")
                        mask[segments == label] = 255
                        if(mask[np.int(img.shape[0]/2),np.int(img.shape[1]/2)]==255):
                            pixels = cv2.countNonZero(mask)/(mask.shape[0]*mask.shape[1])
                            res = cv2.bitwise_or(img,img,mask = mask)
                            res2= cv2.bitwise_or(LAB,LAB,mask = mask)
                            l_channel,a_channel,b_channel=cv2.split(res2)
                            #print(np.mean(a_channel[a_channel!=0]))
                            imgcl.append(res)

                            seg.append(segments)
                i+=1
        else:
            i=0
            for img in Objects:
                #idxmin
                if (data.iloc[i,3]==sum['Mean'].idxmin()):
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                    LAB= cv2.cvtColor(img,cv2.COLOR_RGB2LAB)
                    imgcl.append(img)
                i+=1
    if both_sizes is False:

        if leaves is True:
            i=0
            for img in Objects:
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                LAB= cv2.cvtColor(img,cv2.COLOR_RGB2LAB)
                if mask_ is False:
                    segments = slic(img, n_segments=10, sigma=3, start_label=1,compactness=20)
                else:
                    segments = slic(img, n_segments=3, sigma=3, start_label=1,compactness=20, mask=mascaras[i])
                i+=1
                for label in np.unique(segments):
                    mask = np.zeros((img.shape[0],img.shape[1]), dtype="uint8")
                    mask[segments == label] = 255
                    if(mask[np.int(img.shape[0]/2),np.int(img.shape[1]/2)]==255):
                        pixels = cv2.countNonZero(mask)/(mask.shape[0]*mask.shape[1])
                        res = cv2.bitwise_or(img,img,mask = mask)
                        res2= cv2.bitwise_or(LAB,LAB,mask = mask)
                        plt.imshow(res)

                        #print(np.mean(a_channel[a_channel!=0]))
                        imgcl.append(res)
                        seg.append(segments)
        else:
            for img in Objects:
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                LAB= cv2.cvtColor(img,cv2.COLOR_RGB2LAB)
                imgcl.append(img)


    #The measures will be calculated only to the outside fruit
    shape_measures = pd.DataFrame()
    i=0

    for Example in imgcl:
        i+=1
        imgray = cv2.cvtColor(Example, cv2.COLOR_BGR2GRAY)
        LAB= cv2.cvtColor(Example,cv2.COLOR_BGR2LAB)
        blur = cv2.medianBlur(imgray,3,0)
        ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((16,16), np.uint8)
        th3 = cv2.morphologyEx(th3, cv2.MORPH_OPEN,kernel)
        res2= cv2.bitwise_or(LAB,LAB,mask =th3)
        l_channel,a_channel,b_channel =cv2.split(res2)
        MA=np.mean(a_channel[a_channel!=0])
        ML=np.mean(l_channel[l_channel!=0])
        MB=np.mean(b_channel[b_channel!=0])

        '''
        th3 = cv2.dilate(th3, None, iterations=4)
        th3 = cv2.erode(th3, None, iterations=8)
        '''

        contours, hierarchy  = cv2.findContours(th3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #looking for the maximum area if there are more than one contour
        if len(contours)>1:
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            c=contours[max_index]
        else:
            c = contours[0]
         # compute the bounding box for the contour
        (x, y, w, h) = cv2.boundingRect(c)

        area=cv2.contourArea(c)
        epsilon = 0.1*cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,epsilon,True)
        x,y,w,h = cv2.boundingRect(c)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = float(area)/hull_area
        multip=Example.shape[0]*Example.shape[1]
        AreaFruit=(((area/(multip))*((np.pi)*(23.25/2)**2)))/(45339.0/(multip))
        perimeter = cv2.arcLength(c,True)
        circularity= 4*np.pi*(area/perimeter**2)
        boundingRect=h/w
        EL=cv2.fitEllipse(c)
        EllipseRatio=EL[1][0]/EL[1][1]

        Cropped=th3[(y+np.int(0.75*h)):(y+np.int(0.75*h)+1),:]
        width_at_75_h = cv2.countNonZero(Cropped)

        Cropped2=th3[(y+np.int(0.25*h)):(y+np.int(0.25*h)+1),:]
        width_at_25_h = cv2.countNonZero(Cropped2)

        if(width_at_75_h<width_at_25_h):
            auxiliar=width_at_75_h
            width_at_75_h=width_at_25_h
            width_at_25_h=auxiliar
        Cropped3=th3[(y+np.int(0.5*h)):(y+np.int(0.5*h)+1),:]
        width_at_half_h = cv2.countNonZero(Cropped3)

        try:
            b={
            'Individuo': f,
            'Rep': i,
            'height': h*0.026458333,
            'width': w*0.026458333,
            'widht_at_75h':width_at_75_h*0.026458333,
            'widht_at_25h':width_at_25_h*0.026458333,
            'widht_at_half_h':width_at_half_h*0.026458333,
            'Area': area,
            'Perimeter':perimeter,
            'Solidiy':solidity,
            'AreaReal':AreaFruit,
            'circularity':circularity,
            'EllipseRatio':EllipseRatio,
            'l_channel':ML,'a_channel':MA,
            'b_channel':MB}

            temp = pd.DataFrame(b,index=[0])
            shape_measures= pd.concat([shape_measures, temp])
        except:
            continue

    self.Linear_Descriptors=shape_measures
    self.imgcl=imgcl
    self.segment=seg

    return(self)
