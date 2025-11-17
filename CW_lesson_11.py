import pandas as pd #для працювання з csv таблицями
import numpy as np #математична бібліотека
import tensorflow as tf #створює нейронку
from tensorflow import keras #працює з шарами нейронки, частина tensorflow
from tensorflow.keras import layers #теж працює з шарами (може бути мінімум 3 шари, для входу, тренування, вихід)
from sklearn.preprocessing import LabelEncoder #перетворює текстові мітки в числа
import matplotlib.pyplot as plt #для побудови графіків.Треба знати!

df = pd.read_csv('data/figures.csv')
# print(df.head())

encoder = LabelEncoder()
df["Label_enc"] = encoder.fit_transform(df["label"]) #перетворює назви на числа

#вибираємо стопці для навчання
X = df[["area", "perimeter", "corners"]]
y = df[["Label_endc"]]

#створюємо модель
model = keras.Sequential([
    layers.Dense(8, activation = "relu", input_shape = (3,)),
    layers.Dense(8, activation = "relu"),
    layers.Dense(8, activation = "softmax") #тут дз
])

#компіляція моделі. Визначаємо як мережа буде навчатися
model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"]) #1 - підбирає який краще використати алгоритм, 2 - втрати, 3 - точність

#навчання
history = model.fit(X, y, epochs = 300, verbose = 0) #дз

#візуалізація (графіки)
plt.plot(history.history["loss"], label = "loss")
plt.plot(history.history["accuracy"], label = "accuracy")
plt.xlabel("epoch")
plt.ylabel("znach")
plt.title("process of learning the model")
plt.legend()
plt.show()

#тестування
test = np.array(([25, 20, 0]))
pred = model.predict(test)
print(pred)
print(f"model has decided {encoder.inverse_transform([np.argmax])}")