import cv2
import numpy as np
from matplotlib import pyplot as plt

def saltPepperNoise(image):
    row,col,ch = image.shape
    s_vs_p = 0.5
    amount = 0.04
    noisy = np.copy(image)
    num_salt = int(np.ceil(amount*image.size*s_vs_p))
    corrds = [np.random.randint(0,i-1,num_salt) for i in image.shape]
    noisy[corrds] = 1
    num_pep = int(np.ceil(amount*image.size*s_vs_p))
    corrds = [np.random.randint(0,i-1,num_pep) for i in image.shape]
    noisy[corrds] = 0
    return noisy
    
img = cv2.imread("note.png")
img = img/255
noise_img = saltPepperNoise(img)
cv2.imshow("saltPepperNoise",noise_img)
cv2.waitKey(0)

img = cv2.imread("note.png")
hist,bins = np.histogram(img.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()
plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()