#!/usr/bin/env python
# coding: utf-8

# In[19]:


def color_extract(PATH, LOWER, UPPER):
    # PATH: image path to read it
    #LOWER: Lower range of the color which need to extract from image
    #UPPER: Upper range of the color which need to extract from image
    
    import cv2
    import os
    import numpy as np
    
    # Read image
    img= cv2.imread(PATH)
    
    # Image is blurred
    blurred = cv2.GaussianBlur(img, (11,11), 0)
    
    #BGR to HSV conversion of the image
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    #obtain the grayscale image of the original image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Lower and upper range of the color
    lower = np.array(LOWER)
    upper = np.array(UPPER)
    
    #create a mask using the bounds set
    mask = cv2.inRange(hsv,lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    #create an inverse of the mask
    mask_inv = cv2.bitwise_not(mask)
    
    #Filter the regions containing colours other than color from the grayscale image(background)
    background = cv2.bitwise_and(gray, gray, mask = mask_inv)
    
    #Filter only the color from the original image using the mask(foreground)
    output = cv2.bitwise_and(img, img, mask=mask)

    # Display the resulting frame
    cv2.imshow('out1',mask)
    cv2.imshow('out2',mask_inv)
    cv2.imshow('out3',background)
    cv2.imshow('out',output)
    cv2.waitKey(0)


    cv2.destroyAllWindows()


# In[20]:


# extract orange color
color_extract(r'D:\Winjit_training\hardhat_detection\accuracy_imp\Orange\img_210.jpg',
             [6, 60, 168], 
             [20, 255, 255])


# In[ ]:




