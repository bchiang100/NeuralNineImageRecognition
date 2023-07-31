import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images / 255, testing_images / 255

class_names = ['Plane' ,'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])

plt.show()

training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

model = models.Sequential()
# convolutional layers filter for features in an image
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
# max pooling layers reduces the image to the essential information 
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
# flattens matrix to 1 x X layer
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
# softmax means scaling all the results so they are percentages and adding up to 1 to obtain distrubutions of probabilities
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# epoch are how often the model if going to see the same data over and over again
model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

loss, accuracy = model.evaluate(testing_images, testing_labels)
# how wrong our model is
print(f"Loss: {loss}")
# how much percent of the testing examples were classified correctly
print(f"Accuracy: {accuracy}")

model.save(('image_recognition.model'))

