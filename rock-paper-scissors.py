import numpy as np
import os
from random import choice
from tensorflow.keras.models import load_model
from preprocessandtrain import preprocess
import cv2
REV_CLASS_MAP = {
    0: "empty",
    1: "rock",
    2: "paper",
    3: "scissors"
}

def mapper(value):
    return REV_CLASS_MAP[value]

IMG_SHAPE = (225, 225)

def find_winner(user_move, computer_move):
    pass

def main():
    model = load_model("rock-paper-scissors-model.h5")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error opening video")

    prev_move = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.rectangle(frame, (75, 75), (300, 300), (0, 0, 255), 2)

        capture_region = frame[75:300, 75:300]
        img = cv2.cvtColor(capture_region, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SHAPE)

        pred = model.predict(np.array([img]))
        user_move = mapper(np.argmax(pred[0]))

        winner = None
        computer_move = None
        if prev_move != user_move:
            if user_move != 'empty':
                computer_move = choice(['rock','paper','scissor'])
                winner = find_winner(user_move, computer_move)
            else:
                computer_move = 'empty'
                winner = 'waiting...'

        prev_move = user_move

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "Your Move: " + user_move, (50, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Rock Paper Scissors", frame)

        k = cv2.waitKey(10)
        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

main()