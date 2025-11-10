import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def generate_image(color, shape):
    img = np.zeros((200, 200, 3), np.uint8)
    if shape == 'circle':
        cv2.circle(img, (100, 100), 50, color, -1)
    elif shape == 'square':
        cv2.rectangle(img, (50, 50), (150, 150), color, -1)
    elif shape == 'triangle':
        points = np.array([[100, 40], [40, 160], [160, 160]])
        cv2.drawContours(img, [points], 0, color, -1)
    return img

X = []
y = []
colors = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'yellow': (0, 255, 255),
    'magenta': (255, 0, 255),
    'cyan': (255, 255, 0),
    'purple': (128, 0, 128),
    'orange': (0, 165, 255)
}
shapes = ['circle', 'square', 'triangle']

NUM_SAMPLES_PER_CLASS = 100

for color_name, bgr in colors.items():
    for shape in shapes:
        for _ in range(NUM_SAMPLES_PER_CLASS):
            img = generate_image(bgr, shape)
            mean_color = cv2.mean(img)[:3]
            features = [mean_color[0], mean_color[1], mean_color[2]]
            X.append(features)
            y.append(f"{color_name}_{shape}")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Точність моделі (з {NUM_SAMPLES_PER_CLASS} прикладів, k=7):", round(accuracy * 100, 2), "%")

recent_predictions = []
def get_smoothed_prediction(new_prediction):
    recent_predictions.append(new_prediction)
    counts = {}
    for pred in recent_predictions:
        counts[pred] = counts.get(pred, 0) + 1

    max_count = 0
    smoothed_pred = None
    for pred, count in counts.items():
        if count > max_count:
            max_count = count
            smoothed_pred = pred
    return smoothed_pred


test_color_bgr = (0, 255, 255)
test_shape = 'square'
expected_label = f"yellow_{test_shape}"

test_img = generate_image(test_color_bgr, test_shape)
mean_color = cv2.mean(test_img)[:3]

prediction_result = model.predict([mean_color])[0]
print("---")
print(f"Тест: Очікуємо {expected_label}")
print("Передбачення (один кадр):", prediction_result)

distances, indices = model.kneighbors([mean_color])
nearest_neighbors_labels = [y_train[i] for i in indices[0]]

neighbor_votes = {}
for label in nearest_neighbors_labels:
    neighbor_votes[label] = neighbor_votes.get(label, 0) + 1

probabilities = {label: count / model.n_neighbors for label, count in neighbor_votes.items()}
sorted_probabilities = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)

print(f"Ймовірності (за {model.n_neighbors} сусідами):")
for label, prob in sorted_probabilities:
    print(f"  {label}: {round(prob * 100, 1)}%")

simulated_predictions = [
    f"yellow_square",
    f"yellow_square",
    f"yellow_square",
    f"orange_circle",
    f"yellow_square"
]

for i, pred in enumerate(simulated_predictions):
    smoothed = get_smoothed_prediction(pred)
    print(f"Кадр {i + 1}: Передб. - {pred:<15} | Згладж. - {smoothed}")

text = f"Pred: {prediction_result}"
prob_text = f"Prob: {round(probabilities.get(prediction_result, 0) * 100, 1)}%"

cv2.putText(test_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.putText(test_img, prob_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

cv2.imshow("Test image", test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()