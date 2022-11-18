# Median Spatial Domain Filtering 

'''
Median Filtering: It is also known as nonlinear filtering. 
It is used to eliminate salt and pepper noise. 
Here the pixel value is replaced by the median 
value of the neighboring pixel. 

'''

import cv2
import matplotlib.pyplot as plt 
import math
import numpy as np


# Read the image 
img_noisy1 = cv2.imread('note.png',0) 

# Obtain the number of rows and columns 
# of the image 
m, n = img_noisy1.shape 

# Traverse the image. For every 3X3 area, 
# find the median of the pixels and 
# replace the ceter pixel by the median 
img_new1 = np.zeros([m, n]) 

for i in range(1, m-1): 
	for j in range(1, n-1): 
		temp = [img_noisy1[i-1, j-1], 
			img_noisy1[i-1, j], 
			img_noisy1[i-1, j + 1], 
			img_noisy1[i, j-1], 
			img_noisy1[i, j], 
			img_noisy1[i, j + 1], 
			img_noisy1[i + 1, j-1], 
			img_noisy1[i + 1, j], 
			img_noisy1[i + 1, j + 1]] 
		
		temp = sorted(temp) 
		img_new1[i, j]= temp[4] 

img_new1 = img_new1.astype(np.uint8) 



# Display the images
plt.imshow(img_noisy1) 
plt.show()

# Display the images
plt.imshow(img_new1) 
plt.show()