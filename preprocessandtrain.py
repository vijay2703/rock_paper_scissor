import os
import cv2
from model import generate_model

TRAINING_DATA_DIR = "C:\\Users\\vijay\\PycharmProjects\\opencv_begin\\venv\\training_data"

def preprocess(img):
    width = 225
    height = 225
    dimensions = (width, height)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dimensions, interpolation=cv2.INTER_CUBIC)
    return img

def get_dataset():
    dataset = []
    for label_dir in os.listdir(TRAINING_DATA_DIR):
        path = os.path.join(TRAINING_DATA_DIR, label_dir)
        if not os.path.isdir(path):
            continue
        for image_file in os.listdir(path):
            img = cv2.imread(os.path.join(path, image_file))
            img = preprocess(img)
            dataset.append([img, label_dir])


    return zip(*dataset)


def main():
    X, y = get_dataset()
    generate_model(X, y)


if __name__ == "__main__":
    main()