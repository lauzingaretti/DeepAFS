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
from scipy.stats import norm
from sklearn.neighbors import KernelDensity


#import glob --> I don't need this package

#####################
class imgObjects(load_image):
#####################
    """ This class calculates linear descriptors from a threshold image """
    #th3 from class load_image
    #min_expected_ratio betweeen objects to looking for
    #max_expected_ratio betweeen objects to looking for

    def __init__(self):
        # Invoca al constructor de clase load_image
        return(self)

    def ExtractObjects(self,min_expected_ratio=1, max_expected_ratio=2,output_size=None,write_output=True,output_path=None,output_name=None,label=False,expected_char="IP|LlC-0123456789"):
        """
        self an object from load_image class
        min_expected_ratio: minimum approximate ratio between height and width from fruits/leaves/ objects into the image (1 by default).
        Note that this is only necessary if there are others objects into the image
        max_expected_ratio: maximum approximate ratio between height and width from fruits/leaves/ objects into the image (2 by default).
        output_size: selected size (square) to the output files. If None, the size is extracted from the maximum contours.
        write_output: Would you like to save the extracted images? Default is True.
        output_path: path where the output should be saved. If none, the image would be saved in the main directory.
        output_name: the label for the output name. If none, the label is just "File".
        label: if label is True, the software try to read  automatically the label from the image.  Default is False.
        expected_char: some of char expected into the label, it helps to increase the pytesseract precision. For instance, if all the labels are in the format "Img-xxx",
        being xxx a numeric expression expected_char would be "|Img-0123456789".
        """
        mask=self.mask
        Example=self.img
        Example2= cv2.cvtColor(Example, cv2.COLOR_BGR2GRAY)
        col=(np.mean(Example2))
        col= col.astype(int)

        if output_name is None and label is False:
            output_name="File"

        if output_path is None:
            output_path= os.getcwd()


        if type(mask) is list:
            #corregir esto
            contours, hierarchy  = cv2.findContours(mask[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours, hierarchy  = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h1=[]
        w1=[]
        q3=[]

        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            h1.append(h)
            w1.append(w)
            q3.append(h/w)

        q3=np.asarray(q3)
        h1=np.asarray(h1)
        w1=np.asarray(w1)

        #select the proper contours
        l1=set(list(np.where(h1>=100)[0])).intersection(list(np.where(w1>=100)[0]))
        az=np.sort(np.unique(sum([list(np.where(q3<=5)[0]),list(np.where(q3>0.2)[0])],[])))
        az=list(set(az) & set(l1))

        # select the image size
        a1=np.max(h1[az])
        a1=a1.astype(int)
        a2=np.max(w1[az])
        a2=a2.astype(int)
        anchos=np.max([a1,a2])


        if output_size is None:
            anchos=anchos.astype(int)
        else:
            if output_size<anchos:
                print("warning: selected output_size is too small")
                anchos=anchos.astype(int)
            if output_size>=anchos:
                anchos=output_size


        # if ratio is None, then any ratio is allowed
        if min_expected_ratio is None:
            min_expected_ratio=0
        if max_expected_ratio is None:
            max_expected_ratio=2000000

        #reading the image label if present
        if label is True:
            new_img1=[]
            for c in contours:
                x,y,w,h = cv2.boundingRect(c)

                if w>100 and h>100 and h/w>0.1 and h/w<0.6:
                    #the label is expected to be a rectangle with width 3 4 times  higher than height
                	new_img1=Example[y:y+h,x:x+w]

            if type(new_img1) is list:
                output_name="File"

            if type(new_img1) is not list:
                gray=cv2.cvtColor(new_img1,cv2.COLOR_BGR2GRAY)
                thresh = cv2.threshold(gray, 10, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,np.ones((8,8), np.uint8))
                #erode because the image is inverted then
                thresh = cv2.erode(thresh, np.ones((3,3),np.uint8), iterations=1)
                coords = np.column_stack(np.where(thresh > 0))
                angle = cv2.minAreaRect(coords)[-1]
                if angle < -45:
                	angle = -(270 + angle)

            # otherwise, just take the inverse of the angle to make
            # it positive
                else:
                    angle = -angle


                (h, w) = thresh.shape[:2]
                center = (w // 2, h // 2)

                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                ref = cv2.warpAffine(thresh, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


                ref=ref[int(round(ref.shape[0]/5)):int(round(4*ref.shape[0]/5)),int(round(ref.shape[1]/5)):int(round(4*ref.shape[1]/5))]
                #plt.imshow(255-ref)
                #deleting edges
                ns=str(expected_char)
                custom_config=r'--psm 6 --oem 3 -c tessedit_char_whitelist=' +str(ns)

                text =  pytesseract.image_to_string(255-ref,config=custom_config)
                text_v = pytesseract.image_to_string(imutils.rotate(255-ref,180),config=custom_config)
                #11 --oem 3'
                if (len(text)==0 and len(text_v)==0):
                    print("label not found")
                    output_name="File"
                if(len(text)>len(text_v)):
                    output_name=text
                else:
                    output_name=text_v
                #print(output_name)


        contours2 = [contours[i] for i in az]
        new_img=[]
        all_masks=[]

        for c in contours2:

            x,y,w,h = cv2.boundingRect(c)

            #if h/Example.shape[0]>np.mean(h1) and w/Example.shape[1]>np.mean(w1)and h/w> min_expected_ratio and h/w<= max_expected_ratio:
            if  h/w> min_expected_ratio and h/w<= max_expected_ratio:
                #print(h/w)
                #print(w/Example.shape[0])

                img=Example[y:y+h,x:x+w]
                m2=mask[y:y+h,x:x+w]
                ht, wd, cc= img.shape

                # create new image of desired size and color (blue) for padding
                ww = anchos
                hh = anchos
                if col < 255/2:
                    color = (0,0,0)
                else:
                    color=(255,255,255)
                result = np.full((hh,ww,cc), color, dtype=np.uint8)
                result2 = np.full((hh,ww,cc), color, dtype=np.uint8)
                result2 = cv2.cvtColor(result2,cv2.COLOR_BGR2GRAY)
                # compute center offset
                xx = (ww - wd) // 2
                yy = (hh - ht) // 2

                # copy img image into center of result image
                result[yy:yy+ht, xx:xx+wd] = img
                result2[yy:yy+ht, xx:xx+wd] = m2
                new_img.append(result)
                all_masks.append(result2)

        self.objects=new_img
        self.all_masks=all_masks
        self.id=output_name
        if write_output is True:
            index=0
            for c in new_img:
                index+=1
                if os.getcwd().startswith("/"):
                    cv2.imwrite(str(output_path) +  '/' + str(output_name) + '_' + str(index) +'.png', c )
                else:
                    cv2.imwrite(str(output_path) + '\ ' + str(output_name) + '_' + str(index)  + '.png', c )
        return(self)
