from keras.models import load_model
import cv2 as cv
import numpy as np

label_list = [
  "red-ish",
  "green-ish",
  "blue-ish",
  "orange-ish",
  "yellow-ish",
  "pink-ish",
  "purple-ish",
  "brown-ish",
  "grey-ish"
]
def nothing(x):
	pass

model = load_model("my_model.h5")
img = np.zeros((300, 512, 3), np.uint8)
cv.namedWindow("image")

cv.createTrackbar("R", "image", 160, 255, nothing)
cv.createTrackbar("G", "image", 160, 255, nothing)
cv.createTrackbar("B", "image", 160, 255, nothing)

while 1:
	cv.imshow("image", img)
	k = cv.waitKey(1) & 0xFF
	if k == 27:
		break

	r = cv.getTrackbarPos("R", "image")
	g = cv.getTrackbarPos("G", "image")
	b = cv.getTrackbarPos("B", "image")
	img[:] = [b, g, r]
	cv.putText(img, label_list[np.argmax(model.predict(np.array([[r / 255, g / 255, b / 255]])))], (0, 100), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv.LINE_AA)


cv.destroyAllWindows()