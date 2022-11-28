#credit : https://www.freedomvc.com/index.php/2022/01/17/basic-background-remover-with-opencv/
import cv2
import numpy as np
from matplotlib import pyplot as plt
 
#Use this helper function if you are working in Jupyter Lab
#If not, then directly use cv2.imshow(<window name>, <image>)
myimage = "/home/vk/Desktop/CV_BackgroundRemoval/images/3.jpg"
def showimage(myimage):
    if (myimage.ndim>2):  #This only applies to RGB or RGBA images (e.g. not to Black and White images)
        myimage = myimage[:,:,::-1] #OpenCV follows BGR order, while matplotlib likely follows RGB order
         
    fig, ax = plt.subplots(figsize=[10,10])
    ax.imshow(myimage, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

    #method1 : 

    #Based on this, we designed our background remover with the following strategy:

#Perform Gaussian Blur to remove noise
#Simplify our image by binning the pixels into six equally spaced bins in RGB space. In other words convert into a 5 x 5 x 5 = 125 colors
#Convert our image into greyscale and apply Otsu thresholding to obtain a mask of the foreground
#Apply the mask onto our binned image keeping only the foreground (essentially removing the background)

def bgremove1(myimage=myimage):
    myimage =cv2.imread(myimage)
    # Blur to image to reduce noise
    myimage = cv2.GaussianBlur(myimage,(5,5), 0)
 
    # We bin the pixels. Result will be a value 1..5
    bins=np.array([0,51,102,153,204,255])
    myimage[:,:,:] = np.digitize(myimage[:,:,:],bins,right=True)*51
 
    # Create single channel greyscale for thresholding
    myimage_grey = cv2.cvtColor(myimage, cv2.COLOR_BGR2GRAY)
 
    # Perform Otsu thresholding and extract the background.
    # We use Binary Threshold as we want to create an all white background
    ret,background = cv2.threshold(myimage_grey,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
 
    # Convert black and white back into 3 channel greyscale
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    cv2.imwrite("method1_background_image.png",background)

    # Perform Otsu thresholding and extract the foreground.
    # We use TOZERO_INV as we want to keep some details of the foregorund
    ret,foreground = cv2.threshold(myimage_grey,0,255,cv2.THRESH_TOZERO_INV+cv2.THRESH_OTSU)  #Currently foreground is only a mask
    foreground = cv2.bitwise_and(myimage,myimage, mask=foreground)  # Update foreground with bitwise_and to extract real foreground
    cv2.imwrite("method1_foreground_image.png",foreground)

    # Combine the background and foreground to obtain our final image
    finalimage = background+foreground

    cv2.imwrite("method1_bm.png",finalimage)
 
    return finalimage

#Background Remover with OpenCV – Method 2 – OpenCV2 Simple Thresholding
#Obviously in method 1, we performed a lot of image processing. As can be seen, Gaussian Blur, and Otsu thresholding require a lot of processing. Additionally, when applying Gaussian Blur and binning, we lost a lot of detail in our image. Hence, we wanted to design an alternative strategy that will hopefully be faster. Balanced against efficiency and knowing OpenCV is a highly optimized library, we opted for a thresholding focused approach:

#Convert our image into Greyscale
#Perform simple thresholding to build a mask for the foreground and background
#Determine the foreground and background based on the mask
#Reconstruct original image by combining foreground and background

def bgremove2(myimage=myimage):
    myimage =cv2.imread(myimage)

    # First Convert to Grayscale
    myimage_grey = cv2.cvtColor(myimage, cv2.COLOR_BGR2GRAY)
 
    ret,baseline = cv2.threshold(myimage_grey,127,255,cv2.THRESH_TRUNC)
 
    ret,background = cv2.threshold(baseline,126,255,cv2.THRESH_BINARY)
 
    ret,foreground = cv2.threshold(baseline,126,255,cv2.THRESH_BINARY_INV)
 
    foreground = cv2.bitwise_and(myimage,myimage, mask=foreground)  # Update foreground with bitwise_and to extract real foreground
    cv2.imwrite("method2_foreground_image.png",foreground)
    # Convert black and white back into 3 channel greyscale
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    cv2.imwrite("method2_background_image.png",background)

 
    # Combine the background and foreground to obtain our final image
    finalimage = background+foreground
    cv2.imwrite("method2_image.png",finalimage)
    return finalimage

#Background Remover with OpenCV – Method 3 – Working in HSV Color Space
#Until now, we have been working in BGR color space. With this in mind, our images are prone to poor lighting and shadows. Unquestionably, we wanted to know if working in HSV color space would render better results. In order not to lose image detail we also decided not to perform Gaussian Blur nor image binning. Instead to focus on Numpy for thresholding and generating image masks. Generally, our strategy was as follows:

#Convert our image into HSV color space
#Perform simple thresholding to create a map using Numpy based on Saturation and Value
#Combine the map from S and V into a final mask
#Determine the foreground and background based on the combined mask
#Reconstruct original image by combining extracted foreground and background

def bgremove3(myimage=myimage):
    myimage =cv2.imread(myimage)

    # BG Remover 3
    myimage_hsv = cv2.cvtColor(myimage, cv2.COLOR_BGR2HSV)
     
    #Take S and remove any value that is less than half
    s = myimage_hsv[:,:,1]
    s = np.where(s < 127, 0, 1) # Any value below 127 will be excluded
 
    # We increase the brightness of the image and then mod by 255
    v = (myimage_hsv[:,:,2] + 127) % 255
    v = np.where(v > 127, 1, 0)  # Any value above 127 will be part of our mask
 
    # Combine our two masks based on S and V into a single "Foreground"
    foreground = np.where(s+v > 0, 1, 0).astype(np.uint8)  #Casting back into 8bit integer
 
    background = np.where(foreground==0,255,0).astype(np.uint8) # Invert foreground to get background in uint8
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)  # Convert background back into BGR space
    cv2.imwrite("method3_background_image.png",background)
    foreground=cv2.bitwise_and(myimage,myimage,mask=foreground) # Apply our foreground map to original image
    cv2.imwrite("method3_foreground_image.png",foreground)
    finalimage = background+foreground # Combine foreground and background
    cv2.imwrite("method3_image.png",finalimage)
    return finalimage

bgremove1()
bgremove2()
bgremove3()