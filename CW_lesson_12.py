import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.preprocessing import image

#1 uploading file
train_ds = tf.keras.preprocessing.image_dataset_from_directory("data/train",
                                                               image_size = (128, 128),
                                                               batch_size = 30,
                                                               label_mode = "categorical")
test_ds = tf.keras.preprocessing.image_dataset_from_directory("data/test",
                                                               image_size = (128, 128),
                                                               batch_size = 30,
                                                               label_mode = "categorical")
#2 normalization of a picture
normalization_layer = layers.Rescaling(1./255) #format of the library
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

#3 building a model
model = models.Sequential()
#simple signs
model.add(layers.Conv2D(32, #amount of filters
                        (3, 3), #size of a filters
                        activation = "relu", #function activation
                        input_shape = (128, 128, 3))) #form of an entering image (RGB)
model.add(layers.MaxPooling2D((2, 2))) #reduces card od definitions 2 times
#the deepest signs
model.add(layers.Conv2D(64, (3, 3), activation = "relu"))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation = "relu"))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(64, activation = "relu"))
model.add(layers.Dense(3, activation = "softmax"))

#4 compilation
model.compile(optimizer = "adam",
              loss = "categorical_crossentropy",
              metrics = ["accuracy"])
#5 teaching the model
history = model.fit(train_ds, epochs = 50, validation_data = test_ds)
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc}")
class_name = ["cars", "cats", "dogs"]
img = image.load_img()
image_array = image.img_to_array(img)
image_array = image_array/255
image_array = np.expand_dims(image_array, axis=0)
prediction = model.predict(image_array)
predict_index = np.argmax(prediction)
print(f"Prediction by classes: {prediction[0]}")
print(f"The model has decided: {class_name[predict_index]}")

