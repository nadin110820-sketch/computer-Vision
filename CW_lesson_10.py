import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#1 створюємо функцію для генерації простих фігур
def generate_image(color, shape):
    img = np.zeros((200, 200, 3), np.uint8)
    if shape == "circle":
        cv2.circle(img,(100,100),50, color, -1)
    elif shape == "square":
        cv2.rectangle(img,(50, 50),(150,150),color, -1)
    elif shape == "triangle":
        points = np.array([[100, 40], [40, 160], [160, 100]])
        cv2.drawContours(img, [points], 0, color, -1)
    return img

#2 формуємо набори данних
X = [] #список ознак
y = [] #список міток

colors = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
}
shapes = ["circle", "square", "triangle"]

for color_name, bgr in colors.items():
    for shape in shapes:
        for _ in range(10):
            img = generate_image(bgr, shape)
            mean_color = cv2.mean(img)[:3] #(b, g, r, alpha)
            features = [mean_color[0], mean_color[1], mean_color[2]]
            X.append(features)
            y.append(f"{color_name}_{shape}")
#3 розділяємо дані 70 на 30, 70 - для навчання 30 - для перебірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y) #ознаки для навчання, ознаки для перевірок, правільні відповіді для навчання, правильні відповіді для перевірок

#4 навчаємо модель
model = KNeighborsClassifier(n_neighbors = 3) #бажано ставити не парні числа
model.fit(X_train, y_train)

#5 перевіряємо точність
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {round(accuracy * 100, 2)}%")

test_image = generate_image((0, 255, 0), "circle")
mean_color = cv2.mean(test_image)[:3]
prediction = model.predict([mean_color])
print(f"Prediction: {prediction[0]}")

cv2.imshow("result", test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



