# import pandas as pd  # working with csv tables
# import numpy as np   # mathematics
# import tensorflow as tf  # ai
# from tensorflow import keras  # розширення
# from tensorflow.keras import layers  # РОЗШИРЕННЯ для розширення
# from sklearn.preprocessing import LabelEncoder  # текстові мітки в числа
# import matplotlib.pyplot as plt  # графічно виводити статистику
#
#
# df = pd.read_csv("data/figures_5types_1.csv")
#
#
# encoder = LabelEncoder()
# df['label_enc'] = encoder.fit_transform(df['label'])
#
#
# df['area_perimeter_ratio'] = df['area'] / df['perimeter']
#
# X = df[['area', 'perimeter', 'corners', 'area_perimeter_ratio']]
# y = df['label_enc']
#
#
# num_classes = len(encoder.classes_)
# model = keras.Sequential([
#     layers.Dense(16, activation='relu', input_shape=(4,)),
#     layers.Dense(16, activation='relu'),
#     layers.Dense(num_classes, activation='softmax')
# ])
#
# model.compile(
#     optimizer="adam",
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )
#
# history = model.fit(X, y, epochs=500, verbose=0)
#
# plt.plot(history.history['loss'], label='Втрати')
# plt.plot(history.history['accuracy'], label='Точність')
# plt.xlabel('Epoch')
# plt.ylabel('Value')
# plt.title('Process learning model')
# plt.legend()
# plt.show()
#
# area = 25
# perimeter = 20
# corners = 0
# ratio = area / perimeter
#
# test = np.array([[area, perimeter, corners, ratio]])
#
# pred = model.predict(test)
# print(f'Імовірність кожного класу: {pred}')
# print(f'Модель визначила: {encoder.inverse_transform([np.argmax(pred)])}')

import pandas as pd  # working with csv tables
import numpy as np   # mathematics
import tensorflow as tf  # ai
from tensorflow import keras  # розширення
from tensorflow.keras import layers  # РОЗШИРЕННЯ для розширення
from sklearn.preprocessing import LabelEncoder  # текстові мітки в числа
import matplotlib.pyplot as plt  # графічно виводити статистику


df = pd.read_csv("data/figures_5types_2.csv")


encoder = LabelEncoder()
df['label_enc'] = encoder.fit_transform(df['label'])


df['area_perimeter_ratio'] = df['area'] / df['perimeter']

X = df[['area', 'perimeter', 'corners', 'area_perimeter_ratio']]
y = df['label_enc']


num_classes = len(encoder.classes_)
model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(4,)),
    layers.Dense(16, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(X, y, epochs=500, verbose=0)

plt.plot(history.history['loss'], label='Втрати')
plt.plot(history.history['accuracy'], label='Точність')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Process learning model')
plt.legend()
plt.show()

area = 25
perimeter = 20
corners = 0
ratio = area / perimeter

test = np.array([[area, perimeter, corners, ratio]])

pred = model.predict(test)
print(f'Імовірність кожного класу: {pred}')
print(f'Модель визначила: {encoder.inverse_transform([np.argmax(pred)])}')