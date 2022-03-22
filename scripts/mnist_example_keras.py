# USAGE
# python mnist_example_keras.py --output ../media/keras_mnist.png

# Import Packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Function to plot trained model loss / accuracy
def plot_loss_accuracy(H, args):
	# Set up plot
	plt.style.use("ggplot")
	plt.figure()
	# Plot
	plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
	plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
	plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
	# Titles / labels
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend()

	# Save to output
	plt.savefig(args["output"])

# Main
if __name__ == '__main__':
	# Parse commandline arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-o", "--output", required=True,
		help="path to output loss plot")
	args = vars(ap.parse_args())

	# Get MNIST dataset...
	print("[INFO] accessing MNIST...")
	((trainX, trainY), (testX, testY)) = mnist.load_data()

	# Reshape
	trainX = trainX.reshape((trainX.shape[0], 28 * 28 * 1))
	testX = testX.reshape((testX.shape[0], 28 * 28 * 1))

	# Normalize to [0, 1]
	trainX = trainX.astype("float32") / 255.0
	testX = testX.astype("float32") / 255.0

	# One-hot encoding of labels
	lb = LabelBinarizer()
	trainY = lb.fit_transform(trainY)
	testY = lb.transform(testY)

	# Label names
	labelNames = [str(x) for x in lb.classes_]

	# Using 784-256-128-10 architecture using Keras
	model = Sequential()
	model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
	model.add(Dense(128, activation="sigmoid"))
	model.add(Dense(10, activation="softmax"))

	# Train the model
	print("[INFO] training network...")
	# Using SGD Optimiziation Method
	sgd = SGD(0.01)
	model.compile(loss="categorical_crossentropy", optimizer=sgd,
		metrics=["accuracy"])
	H = model.fit(trainX, trainY, validation_data=(testX, testY),
		epochs=100, batch_size=128)

	# Evaluate
	print("[INFO] evaluating network...")
	predictions = model.predict(testX, batch_size=128)
	print(classification_report(testY.argmax(axis=1),
		predictions.argmax(axis=1), target_names=labelNames))

	# Plot training loss / accuracy
	plot_loss_accuracy(H, args)
