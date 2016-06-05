import numpy as np
import cv2
import math
from sklearn import decomposition
from sklearn.feature_extraction import image
from skimage.util.shape import view_as_blocks


def H(x):
    if x>0:
            return 1
    else:
            return 0

def histogramPatch(patch,noOfFilter):  
     numberOfBins = int(math.pow(2,noOfFilter))
     #plt.hist(patch, bins = [i  for i in range(numberOfBins)])    
     ##pending: display histogram here
     return np.histogram(patch, bins = [i  for i in range(numberOfBins)])

##FUNCTION TO DO THE FILTER
def PCAFilter(images, patches_size_x, patches_size_y,noOfFilter):
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
        
        ##TODO: print filters as images to folder
        
        #apply filter
        filtered_images = []
        for k in range(len(images)):
                img = images[k]
                for i in range(len(filters)):
                        filter = filters[i,:]
                        filter = np.reshape(filter, (patches_size_y, patches_size_x))
                        filtered_images.append(cv2.filter2D(img,-1,filter))
                        #TODO: print filtered image  to folder
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
                                #TODO: print filtered image  to folder
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
                        #TODO: display histogram
                        imageHistogram.append(histogram)
                histograms.append(imageHistogram)
        return histograms

