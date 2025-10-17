#!/usr/bin/env python
# coding: utf-8

# In[1]:


# NAME : SARANYA S
# REG.NO: 212223220101


# In[2]:


import cv2
import matplotlib.pyplot as plt


# In[3]:


# Read the Image and convert to grayscale

image=cv2.imread('nature.jpg')
gray_img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)


# In[4]:


# Original image

plt.subplot(2,2,1)
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')


# In[5]:


# Use Global thresholding to segment the image

_,global_thresholded = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)


# In[6]:


# Use Adaptive thresholding to segment the image

adaptive_thresholded = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


# In[7]:


# Use Otsu's method to segment the image 

_,otsu_thresholded = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


# In[8]:


# Global Thresholding


plt.subplot(2, 2, 2)
plt.imshow(global_thresholded, cmap='gray')
plt.title("Global Thresholding")
plt.axis('off')


# In[9]:


# Adaptive Thresholding

plt.subplot(2, 2, 3)
plt.imshow(adaptive_thresholded, cmap='gray')
plt.title("Adaptive Thresholding")
plt.axis('off')


# In[10]:


# Otsu's Method

plt.subplot(2, 2, 4)
plt.imshow(otsu_thresholded, cmap='gray')
plt.title("Otsu's Method")
plt.axis('off')


# In[11]:


# Show the plot

plt.tight_layout()
plt.show()


# In[ ]:




