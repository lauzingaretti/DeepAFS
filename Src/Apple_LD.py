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


def Apple_LD(self, both_sizes=False):
    """
    This function remove the calix for apple image prior to compute the linear descriptors.
    it is only adapted for apples images
    both_sizes: logical, if true, it removes outer apple prior to calculate LD
    """
    Objects =self.objects
    shape= self.shape
    mascaras=self.all_masks
    f=self.id
    height=shape[0]
    width=shape[1]
    seg=[]
    imgcl=[]
    shape_measures = pd.DataFrame()
    kk=0
    Objects2=[]


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

        i=0
        for img in Objects:
            #idxmin
            if (data.iloc[i,3]==sum['Mean'].idxmax()):
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                LAB= cv2.cvtColor(img,cv2.COLOR_RGB2LAB)
                Objects2.append(img)
            i+=1
    if len(Objects2)==0:
        Objects2=Objects
        del Objects

    for im in Objects2:
        kk+=1
        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        pixel_values = im.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        centroid_initialization_strategy = cv2.KMEANS_RANDOM_CENTERS

        _, labels, centers = cv2.kmeans(pixel_values,
                                    5,
                                    None,
                                    stop_criteria,
                                    10,
                                    centroid_initialization_strategy)

        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]

        segmented_image = segmented_data.reshape(im.shape)
        m=[]
        for i in np.unique(labels):
            m.append(np.count_nonzero(labels.flatten()==i)/(im.shape[0]*im.shape[1]))
        aux=list(np.argsort(m))
        aux.remove(np.argmin(m))
        aux.remove(np.argmax(m))

        ind_hearth= np.argsort(m)[1]
        ind_triangle= np.argsort(m)[0]

        mesocarp_= list([np.argsort(m)[2], np.argsort(m)[3]])


        mask= np.zeros((im.shape[0],im.shape[1]),dtype="uint8")
        La=labels.reshape(im.shape[0],im.shape[1])
        mask[La== ind_triangle] = 255
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask=cv2.medianBlur(mask,25,0)

        contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #print(len(contours))
        calix_a=0
        stem_pit_a=0

        for c in contours:
            area=cv2.contourArea(c)
            (x, y, w, h) = cv2.boundingRect(c)

            if (y>=im.shape[0]/2):
                if (h/w>=0.2):
                    calix_a=calix_a+area
            else:
                if (h/w>=0.2):
                    stem_pit_a=stem_pit_a + area
        multip=im.shape[0]*im.shape[1]
        calix_a=(((calix_a/(multip))*((np.pi)*(23.25/2)**2)))/(45339.0/(multip))
        stem_pit_a=(((stem_pit_a/(multip))*((np.pi)*(23.25/2)**2)))/(45339.0/(multip))

        masc2= np.zeros((im.shape[0],im.shape[1]),dtype="uint8")

        for cluster in mesocarp_:
            mask= np.zeros((im.shape[0],im.shape[1]),dtype="uint8")
            La=labels.reshape(im.shape[0],im.shape[1])
            mask[La == cluster] = 255
            masc2=mask+masc2

        kernel = np.ones((5,5), np.uint8)
        masc2= cv2.morphologyEx(masc2, cv2.MORPH_OPEN, kernel)
        masc2=cv2.medianBlur(masc2,25,0)
        contours, hierarchy = cv2.findContours(masc2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        mes_area=0
        for c in contours:
            area=cv2.contourArea(c)
            mes_area=area+mes_area

        mes_area=(((mes_area/(multip))*((np.pi)*(23.25/2)**2)))/(45339.0/(multip))



        #medidas normales
        masc= np.zeros((im.shape[0],im.shape[1]),dtype="uint8")

        for cluster in aux:
            mask= np.zeros((im.shape[0],im.shape[1]),dtype="uint8")
            La=labels.reshape(im.shape[0],im.shape[1])
            mask[La == cluster] = 255
            masc=mask+masc



        masc=cv2.medianBlur(masc,25,0)
        kernel = np.ones((15,15), np.uint8)
        masc= cv2.morphologyEx(masc, cv2.MORPH_CLOSE, kernel)
        res = cv2.bitwise_or(im,im,mask = masc)
        LAB= cv2.cvtColor(res,cv2.COLOR_BGR2LAB)
        l_channel,a_channel,b_channel =cv2.split(res)
        MA=np.mean(a_channel[a_channel!=0])
        ML=np.mean(l_channel[l_channel!=0])
        MB=np.mean(b_channel[b_channel!=0])

        contours, hierarchy = cv2.findContours(masc,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
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
        multip=im.shape[0]*im.shape[1]
        AreaFruit=(((area/(multip))*((np.pi)*(23.25/2)**2)))/(45339.0/(multip))
        perimeter = cv2.arcLength(c,True)
        circularity= 4*np.pi*(area/perimeter**2)
        boundingRect=h/w
        EL=cv2.fitEllipse(c)
        EllipseRatio=EL[1][0]/EL[1][1]

        Cropped=masc[(y+np.int(0.75*h)):(y+np.int(0.75*h)+1),:]
        width_at_75_h = cv2.countNonZero(Cropped)

        Cropped2=masc[(y+np.int(0.25*h)):(y+np.int(0.25*h)+1),:]
        width_at_25_h = cv2.countNonZero(Cropped2)

        if(width_at_75_h<width_at_25_h):
            auxiliar=width_at_75_h
            width_at_75_h=width_at_25_h
            width_at_25_h=auxiliar
        Cropped3=masc[(y+np.int(0.5*h)):(y+np.int(0.5*h)+1),:]
        width_at_half_h = cv2.countNonZero(Cropped3)


        imgcl.append(res)
        seg.append(segmented_image)

        try:
            b={
            'Individuo': f,
            'Rep': kk,
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
            'b_channel':MB,
            'calix_a':calix_a,
            'stem_pit_a': stem_pit_a,
            'mesocarp_area':mes_area}

            temp = pd.DataFrame(b,index=[0])
            shape_measures= pd.concat([shape_measures, temp])
        except:
            continue

    self.Linear_Descriptors=shape_measures
    self.imgcl=imgcl
    self.segment=seg


    return(self)


    #cambiar la forma analizar con los colores o ver cómo pensar
