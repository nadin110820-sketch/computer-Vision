import cv2
import os
import shutil

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(PROJECT_DIR, 'images')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
OUT_DIR = os.path.join(PROJECT_DIR, 'out')
PEOPLE_DIR = os.path.join(OUT_DIR, 'people')
NO_PEOPLE_DIR = os.path.join(OUT_DIR, 'no_people')
os.makedirs(PEOPLE_DIR, exist_ok=True)
os.makedirs(NO_PEOPLE_DIR, exist_ok=True)
PROTOTXT_PATH = os.path.join(MODELS_DIR, 'MobileNetSSD_deploy.prototxt')
MODEL_PATH = os.path.join(MODELS_DIR, 'MobileNetSSD_deploy.caffemodel')
net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]
PERSON_CLASS_ID = CLASSES.index("person")
CONF_THRESHOLD = 0.2
def detect_people(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        0.007843,
        (300, 300),
        (127.5, 127.5, 127.5)
    )
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        class_id = int(detections[0, 0, i, 1])
        if class_id == PERSON_CLASS_ID and confidence > CONF_THRESHOLD:
            box = detections[0, 0, i, 3:7]
            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)
            boxes.append((x1, y1, x2, y2, confidence))
    return boxes
allowed_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
files = os.listdir(IMAGES_DIR)
for file in files:
    if not file.lower().endswith(allowed_extensions):
        continue
    img_path = os.path.join(IMAGES_DIR, file)
    img = cv2.imread(img_path)

    if img is None:
        continue
    people_boxes = detect_people(img)
    people_count = len(people_boxes)
    boxed = img.copy()
    for (x1, y1, x2, y2, conf) in people_boxes:
        cv2.rectangle(boxed, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(
        boxed,
        f"People count: {people_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    if people_count > 0:
        shutil.copy(img_path, os.path.join(PEOPLE_DIR, file))
        cv2.imwrite(os.path.join(PEOPLE_DIR, "boxed_" + file), boxed)
    else:
        shutil.copy(img_path, os.path.join(NO_PEOPLE_DIR, file))