## DeepAFS: A python pipeline for automatic fruit segmentation

#### [Laura M Zingaretti](https://publons.com/researcher/3104357/maria-laura-zingaretti/)
m.lau.zingaretti@gmail.com

### Citation

- Zingaretti, L. M., Monfort, A., & Perez-Enciso, M. (2020). Automatic fruit morphology phenome and genetic analysis: An application in the octoploid strawberry. Plant Phenomics (2021). Accepted. 
---
Automatizing phenotype measurement is needed to increase plant breeding efficiency. Morphological traits are relevant in many fruit breeding programs, as appearance influences consumer preference. Often, these traits are manually or semi-automatically obtained. Yet, fruit morphology evaluation can be boosted by resorting to fully automatized procedures and digital images provide a cost- effective opportunity for this purpose. Images are an inexpensive and versatile source of data in agriculture, however analyzing them remains a difficult task, partly due to their mathematical complexities and the scarcity of software. Here, we present **DeepAFS: a python based pipeline for automatic analysis of images fruits**, an automatized pipeline for comprehensive phenomic and genetic analysis of morphology traits extracted from inner and outer fruit images. The pipeline segments, classifies and labels the images, extracts conformation features, including linear (area, perimeter, height, width, circularity, shape descriptor, ratio between height and width) and multivariate (Fourier Elliptical components and Generalized Procrustes) statistics. Inner color patterns are obtained using an autoencoder to smooth out the image. In addition, we develop a variational autoencoder to automatically detect the most likely number of underlying shapes. Bayesian modeling is employed to estimate both additive and dominant effects for all traits. As expected, conformational traits are clearly heritable. Interestingly, dominance variance is higher than the additive component for most of the traits. 

DeepAFS is not the first available tool to  asses fruit shape. Tomato Analyzer (TA) [1] is a tool developed to scan tomatoes, and can be easily adapted to alternative fruits analysis or even roots. However, its main difficulty lies in it is manual and requires many interventions from users, allowing only one image at a time. ImageJ is another powerful tool for image analysis, which has many specific developments on the geometric morphometric field [2], but it lacks the advantages of python. Although it is an open source language, which supports macros and  extensions, it usually requires many actions by users, making it less flexible.   

DeepAFS takes advantage of the OpenCV [3] and scikit-image [4] tools to bring a fast, flexible and extensible tool, which can be used to fruit shape analysis from a Geometric-Morphometric (GM) view [5]. 

The pipeline 1) converts the raw data (fruit images) into a processed curated database, and 2) runs a workflow to automatically analyze fruit shape phenome, by returning not only the fruit linear descriptors, but also landmarks, ready to carry out a multivariate analysis.

### Requirements
See [requirements.txt](https://github.com/lauzingaretti/DeepAFS/blob/master/requirements.txt).

### Installation
Clone this repository

Install requirements.txt -->

Open bash, go to the  main repository folder and run the following command:

      pip install -r requirements.txt

then in python run:

      from Src import load_image
      from Src import scanIm
      from Src import linearDesc
      from Src import Apple_LD
      # alternative modules
      import matplotlib.pyplot as plt
      import numpy as np
      import inspect

Program has been tested in mac and linux only, although it should run as any regular python script in windows.

As in any python project, it is recommended to use a separate environment to avoid conflicts between package versions. You can do that with **conda** as follows:

```
    conda create -n ImageWorkflow
    conda activate ImageWorkflow
    # run python

   ....
    # to finish
    conda deactivate
```
### Quick startup
The basic philosophy of **DeepAFS** is to have a fruit image file with homogeneous background (black/white).The software read the image file (jpg, png, tiff, among others) and extract the features (fruits, leaves)
.


### Examples
The best option to use the program is to follow the examples.
* [`main.ipynb`](https://github.com/lauzingaretti/DeepAFS/blob/master/main.ipynb) illustrates main functionalities.


### Main classes

**DeepAFS** allows storing and accessing image information easily.
*```load_image``` class read the images and allows visualization, histogram visualization and analyses and image segmentation (gray and color based through k-means).

* ```imgObjects``` is suited to take as an input an object  from agis/acis function (automatic gray/ color image segmentation). This function returns the individual objects, masks, label of image (if exists) and discard regions of no interest. The output images can be stored in the software memory or can be save in a predefined folder. The outputs with all the objects are always squared images of a user defined size (default is 1000 px.)

* ```LinearDesc``` is a function that takes as input an  `imgObjects` and calculates different morphological linear descriptors and it also analyzes the color of the outer fruit. The output can be stored as a pandas Data Frame.
 It applies a k-means to automatically separate outer and inner fruits if necessary (i.e. if the  fruits of both sections are in the snapshot). For strawberries, this function also allows to automatically remove the  leaves from the outside fruits. This step is important as leaves are  not part of the fruit shape.

* ```landmarks_gen``` This function takes an output from `imgObjects` and computes a serie of predefined pseudo-landmarks for the fruit contour. This step is necessary to perform the multivariate analyzes.

* ```Apple_LD``` is a function suited to analyze apple image where only fruits with the inner section are present (see more details in the examples). 


### How to contribute
Please send comments, suggestions or report bugs to m.lau.zingaretti@gmail.com

#### Disclaimer
Copyright (C) 2020 Laura M. Zingaretti

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.


---

### Bibliography

[1] Brewer, M. T., Lang, L., Fujimura, K., Dujmovic, N., Gray, S., & van der Knaap, E. (2006). Development of a controlled vocabulary and software application to analyze fruit shape variation in tomato and other plant species. Plant physiology, 141(1), 15-25.

[2] David Legland, Ignacio Arganda-Carreras, Philippe Andrey, MorphoLibJ: integrated library and plugins for mathematical morphology with ImageJ, Bioinformatics, Volume 32, Issue 22, 15 November 2016, Pages 3532–3534, https://doi.org/10.1093/bioinformatics/btw413

[3] Bradski, G., & Kaehler, A. (2008). Learning OpenCV: Computer vision with the OpenCV library. " O'Reilly Media, Inc.".

[4] Van der Walt, S., Schönberger, J. L., Nunez-Iglesias, J., Boulogne, F., Warner, J. D., Yager, N., ... & Yu, T. (2014). scikit-image: image processing in Python. PeerJ, 2, e453.

[5] Klingenberg, C. P. (2010). Evolution and development of shape: integrating quantitative approaches. Nature Reviews Genetics, 11(9), 623-635.
