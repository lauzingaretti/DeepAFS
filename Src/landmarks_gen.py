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
from scipy.interpolate import splprep, splev
from pyefd import elliptic_fourier_descriptors
import warnings
warnings.filterwarnings("ignore")


def landmarks_gen(self,N=50,write_output=False,output_path=None):
    """
    An object from the LinearDesc output
    # N: number of pseudo-landmarks
    # write_output: logical. If true, landmarks are written in output_path. If output_path is None, these are written in the
    current_dir,
    else, landmarks are stored in a list
    """
    files2 = self.imgcl
    id=self.id
    h11=self.imgcl[0].shape[0]
    w11=self.imgcl[0].shape[1]

    landmarks = []
    if write_output is True and output_path is None:
        output_path= os.getcwd()

    index=0

    for x in files2:
        index+=1
        Example = x
        imgray = cv2.cvtColor(Example, cv2.COLOR_RGB2GRAY)
        ret3,th3 = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        contours, hierarchy  = cv2.findContours(th3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)>1:
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            c=contours[max_index]
        else:
            c = contours[0]
         # compute the bounding box for the contour
        (x, y, w, h) = cv2.boundingRect(c)

        img=Example[y:y+h,x:x+w]
        ht, wd, cc= img.shape
        if wd<ht:
            dim = (round((h11/ht)*wd), h11)
        else:
            dim=(h11,h11)
        # resize image
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        ht, wd, cc= img.shape
        # create new image of desired size and color (blue) for padding
        ww = int(h11 + 0.2*h11)
        hh = int(h11 + 0.2*h11)
        color = (0,0,0)
        result = np.full((hh,ww,cc), color, dtype=np.uint8)
        #print(result.shape)
        # compute center offset
        xx = (ww - wd) // 2
        yy = (hh - ht) // 2

        # copy img image into center of result image
        try:
            result[yy:yy+ht, xx:xx+wd] = img
        except:
            continue
        imgray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        ret3,th3 = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        th3 = cv2.medianBlur(th3,15,0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        th3= cv2.morphologyEx(th3, cv2.MORPH_OPEN, kernel, iterations=8)
        th3 = cv2.GaussianBlur(th3,(5,5),0)


        contours, hierarchy  = cv2.findContours(th3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)>1:
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            c=contours[max_index]
        else:
            c = contours[0]
         # compute the bounding box for the contour
        (x, y, w, h) = cv2.boundingRect(c)

        ww = int(h11 + 0.2*h11)
        hh = int(h11 + 0.2*h11)
        cc = 3
        color = (0,0,0)
        res = np.full((hh,ww,cc), color, dtype=np.uint8)

        x,y = c.T
        # Convert from numpy arrays to normal arrays
        x = x.tolist()[0]
        y = y.tolist()[0]
        x=np.r_[x,x[0]]
        y=np.r_[y,y[0]]

        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
        tck, u = splprep([x,y], s=0,per=True)
        # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
        x_new, y_new = splev(np.linspace(0, 1, 1000), tck)
        # Convert it back to numpy format for opencv to be able to display it
        res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new,y_new)]
        smoothened = np.asarray(res_array, dtype=np.int32)
        # Overlay the smoothed contours on the original image

        res=cv2.drawContours(res, smoothened, -1, (255,255,255), 2)
        plt.imshow(res)
        #obtaining mass point
        center=np.zeros((2,),dtype=np.int32)

        m = cv2.moments(c)
        ca=c
        center[0] = int(m["m10"] / m["m00"])
        center[1] = int(m["m01"] / m["m00"])
        #get the left most point
        bottommost = tuple(c[c[:,:,1].argmax()][0])
        center=tuple(center)

        # Get dimensions of the image
        width = res.shape[1]
        height = res.shape[0]

        # Define total number of angles we want


        ht, wd, cc= result.shape
        # create new image of desired size and color (blue) for padding
        ww = int(h11 + 0.2*h11)
        hh = int(h11 + 0.2*h11)
        color = (255,255,255)
        img_white = np.full((hh,ww,cc), color, dtype=np.uint8)


        ref1 = np.zeros_like(img_white)

        points=[]
        for i in range(N):
            # Step #6a
            # Step #6b
            tmp = np.zeros_like(img_white)

            theta = i*(360/N)
            theta *= np.pi/180.0
            # Step #6c
            cv2.line(tmp, (center[0], center[1]),
                   (int(center[0]+np.cos(theta)*width),
                    int(center[1]-np.sin(theta)*height)), (255, 255, 255), 3)
            try:
                x=np.mean(np.nonzero(np.logical_and(tmp, res))[1])
                y=np.mean(np.nonzero(np.logical_and(tmp, res))[0])
                ref1=cv2.circle(ref1, (np.int(x),np.int(y)), radius=2, color=(255,255, 255), thickness=-1)
                points.append((x,y))
            except:
                continue
        df = pd.DataFrame(points,columns = list("xy"))
        if write_output is True:
            if os.getcwd().startswith("/"):
                df.to_csv(str(output_path) + "/" + str(id) + "_landmarks_"+ str(index) + ".csv",index=False)
            else:
                df.to_csv(str(output_path) + '\ ' + str(id) + "_landmarks_"+ str(index) + ".csv",index=False)


        landmarks.append(df)

    self.landmarks=landmarks
    return(self)




def fourier__descriptors(self,order=5, normalize=False):
    """
    To calculate elliptic_fourier_descriptors from a contour
    self: an element of LinearDesc output
    order: The order of Fourier coefficients to calculate.
    normalize I hte coefficients should be normalized
    """
    mascaras = self.imgcl

    Fourier_desc=[]

    for Example in mascaras:
        index+=1
        imgray = cv2.cvtColor(Example, cv2.COLOR_RGB2GRAY)
        ret3,th3 = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        contours, hierarchy  = cv2.findContours(th3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)>1:
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            c=contours[max_index]
        else:
            c = contours[0]
         # compute the bounding box for the contour
        Fourier_desc.append(elliptic_fourier_descriptors(np.squeeze(c), order=order,normalize=normalize))

    self.EFD=Fourier_desc
    return(self)
    
