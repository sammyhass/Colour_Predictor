import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD
import json


FILENAME = "data.json"
with open(FILENAME, "r") as f:
	datastore = json.load(f)

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
colors = []
labels = []

for record in datastore["entries"]:
	col = [record["r"] / 255, record["g"] / 255, record["b"] / 255]
	colors.append(col)
	labels.append(label_list.index(record["label"]))

xs = np.array(colors)
ys = to_categorical(labels)
model = Sequential([
	Dense(20, input_shape=(3, )),
	Activation("sigmoid"),
	Dense(9),
	Activation("softmax")
])
model.compile(optimizer=SGD(lr=0.25), loss=categorical_crossentropy, metrics=["accuracy"])

model.fit(xs, ys, epochs=10)

# test = np.array([[1, 0, 0]])
model.save("my_model.h5")
# print(label_list[np.argmax(model.predict(test))])