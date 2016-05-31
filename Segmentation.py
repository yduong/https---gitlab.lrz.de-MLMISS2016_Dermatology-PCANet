import numpy as np
import cv2
import math
import os
from sklearn import decomposition
from sklearn.feature_extraction import image
from skimage.util.shape import view_as_blocks
from os.path import isfile, join
#import matplotlib.pyplot as plt

##HELPER FUNCTIONS
def VectorizeAndDemean(patch):
        mean = patch.mean()
        return np.apply_along_axis(Substract,0,patch,mean)

def Substract(x,y):
       return x-y

def H(x):
    if x>0:
            return 1
    else:
            return 0

def histogramPatch(patch,noOfFilter):  
     numberOfBins = int(math.pow(2,noOfFilter))
     #plt.hist(patch, bins = [i  for i in range(numberOfBins)])    
     return np.histogram(patch, bins = [i  for i in range(numberOfBins)])

##FUNCTION TO DO THE FILTER
def PCAFilter(images, patches_size_x, patches_size_y,noOfFilter, width, height):
        #get the number of patch, remove mean, vertorized it and store it into the ImagesPatches
        total_element_in_patch = patches_size_x*patches_size_y;
        images_patches = None
        for k in range(len(images)):
            img = images[k]
            #extract patch and reshape
            patches = image.extract_patches_2d(img, (patches_size_y, patches_size_x))
            patches = np.reshape(np.ravel(patches),(len(patches),total_element_in_patch))
            demeanPatches = patches - patches.mean(axis=1)[:, np.newaxis]           
            if images_patches is None:
                images_patches = demeanPatches
            else:
                images_patches = np.concatenate((images_patches, demeanPatches), axis=0)        
        #get the pca
        pca = decomposition.PCA(n_components=noOfFilter)
        pca.fit(images_patches)
        filters = pca.components_
        #apply filter
        filtered_images = []
        for k in range(len(images)):
                img = images[k]
                for i in range(len(filters)):
                        filter = filters[i,:]
                        filter = np.reshape(filter, (patches_size_y, patches_size_x))
                        filtered_images.append(cv2.filter2D(img,-1,filter))
                        #filtered_images.append(scipy.signal.convolve2d(img, filter, mode='same'))

        return filtered_images;

#FUNCTION TO HASH
def hashing(images,number_of_filter1,number_of_filter2,numberOfInitImage):
        hashed_images = [];
        h = np.vectorize(H)
        for k in range(numberOfInitImage):
                startCount = k*number_of_filter1*number_of_filter2
                for i in range(number_of_filter1):
                        hashed_image =math.pow(2,0)*h(images[startCount+i*number_of_filter1])
                        for j in range(1,number_of_filter2):
                                hashed_image = hashed_image + math.pow(2,j-1)*h(images[startCount+i*number_of_filter1 + j])
                        hashed_images.append(hashed_image)
        return hashed_images


#Histogram
def histogramming(images,patches_size_x,patches_size_y, numberOfImage, L1,L2):
        histograms = []
        for i in range(numberOfImage):
                imageHistogram = []
                start = i*numberOfImage
                for k in range(L1):
                        img = images[start+k]
                        blocks = view_as_blocks(img, block_shape=(13, 14))
                        h,w,b1,b2 = blocks.shape
                        blocks = np.ravel(blocks)
                        blocks.shape = (4307,182)
                        histogram = np.apply_along_axis(histogramPatch,1,blocks,L2)
                        imageHistogram.append(histogram)
                histograms.append(imageHistogram)
        return histograms

######### MAIN FUNCTION

imageFiles = [f for f in os.listdir("ISBI2016_ISIC_Part1_Training_Data") if (isfile(join("ISBI2016_ISIC_Part1_Training_Data", f)) and f != ".DS_Store")]
images = []
numberOfInitImage = min(2,len(imageFiles))
for i in range(numberOfInitImage):
    img = cv2.imread("ISBI2016_ISIC_Part1_Training_Data/" + imageFiles[i],0)
    cv2.imshow('image',img) 
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    images.append(img)
    
#initialize the value
height, width = images[0].shape
patches_size_x = 3
patches_size_y = 3
noOfFilter1 = 8
noOfFilter2 = 8
#PCA Filter stage 1
filteredImageStage1 = PCAFilter(images, patches_size_x, patches_size_y,noOfFilter1,width, height)
print("finish stage1")
#for i in range(len(filteredImageStage1)):
#    cv2.imshow('image',filteredImageStage1[i]) 
#    cv2.waitKey(2000)
#    cv2.destroyAllWindows() 
    
#PCA Filter stage 2
filteredImageStage2 = PCAFilter(filteredImageStage1, patches_size_x, patches_size_y,noOfFilter1,width, height)
print("finish stage2")
#for i in range(len(filteredImageStage2)):
#    cv2.imshow('image',filteredImageStage2[i]) 
#    cv2.waitKey(2000)
#    cv2.destroyAllWindows() 

filteredImageStage2 = np.array(filteredImageStage2);
#hashed
hashedImages = hashing([filteredImageStage2[i] for i in range(noOfFilter1*noOfFilter2*numberOfInitImage)],noOfFilter1,noOfFilter2,numberOfInitImage)
for i in range(len(hashedImages)):
    cv2.imshow('image',hashedImages[i]) 
    cv2.waitKey(2000)
    cv2.destroyAllWindows() 
    
#histogram
histogram = histogramming(hashedImages,patches_size_x,patches_size_y,numberOfInitImage,noOfFilter1,noOfFilter2)
