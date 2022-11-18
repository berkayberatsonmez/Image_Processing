import cv2
import matplotlib.pyplot as plt 
import numpy as np

img = cv2.imread('note.png',0)

lst = []
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
         lst.append(np.binary_repr(img[i][j] ,width=24))

eight_bit_img = (np.array([int(i[8]) for i in lst],dtype = np.uint8) * 8).reshape(img.shape[0],img.shape[1])
plt.imshow(eight_bit_img) 
plt.show()
sixteen_bit_img = (np.array([int(i[16]) for i in lst],dtype = np.uint8) * 16).reshape(img.shape[0],img.shape[1])
plt.imshow(sixteen_bit_img) 
plt.show()
twenty_four_bit_img = (np.array([int(i[23]) for i in lst],dtype = np.uint8) * 24).reshape(img.shape[0],img.shape[1])
plt.imshow(twenty_four_bit_img) 
plt.show()