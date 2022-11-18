import cv2
import numpy as np

def adjust_gamma(image, gamma=1.0):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")

	return cv2.LUT(image, table)

original = cv2.imread("note.png")
original = cv2.resize(original, (500, 500))

for gamma in np.arange(3.0, 6.0, 1.0):

	if gamma == 1:
		continue

	gamma = gamma if gamma > 0 else 0.1
	adjusted = adjust_gamma(original, gamma=gamma)
	cv2.putText(adjusted, "g={}".format(gamma), (10, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
	cv2.imshow("input", np.hstack([original, adjusted]))
	cv2.waitKey(0)