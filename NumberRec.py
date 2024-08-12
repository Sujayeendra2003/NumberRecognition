import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# mnist = tf.keras.datasets.mnist
# (X_train, Y_train),(x_test, y_test) = mnist.load_data()

# X_train = tf.keras.utils.normalize(X_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

# model= tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))

# model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(X_train, Y_train, epochs=10)

# model.save('HandWritten.model.keras')
model = tf.keras.models.load_model('HandWritten.model.keras')

image_number = 1
while os.path.isfile(f"Digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"Digits/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"The Number is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!!")
    finally:
        image_number += 1
