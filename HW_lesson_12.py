import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

df = pd.read_csv('data/figures_hw.csv')
encoder = LabelEncoder()
df['label_enc'] = encoder.fit_transform(df['label'])
df['area_perimeter_ratio'] = df['area'] / df['perimeter']
X = df[['area', 'perimeter', 'corners', 'area_perimeter_ratio']]
y = df["label_enc"]

num_classes = len(encoder.classes_)
y_ohe = keras.utils.to_categorical(y, num_classes=num_classes)
model = keras.Sequential([
    layers.Dense(16, activation = 'relu', input_shape = (4,)),
    layers.Dense(8, activation = 'relu'),
    layers.Dense(num_classes, activation = 'softmax')
])
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(X, y_ohe, epochs = 500, verbose = 0)
plt.plot(history.history['loss'], label = 'Loss')
plt.plot(history.history['accuracy'], label = 'Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.show()
test_area = 30.0
test_perimeter = 22.0
test_corners = 4
test_ratio = test_area / test_perimeter

test = np.array([[test_area, test_perimeter, test_corners, test_ratio]])
pred = model.predict(test, verbose=0)
pred_class_index = np.argmax(pred)

print(f'Список усіх класів: {list(encoder.classes_)}')
print(f'Тестові дані: {test[0]}')
print(f'Ймовірність по кожному класу: {pred}')
print(f'Модель визначила: {encoder.inverse_transform([pred_class_index])[0]}')