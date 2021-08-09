import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2)
mpDraw = mp.solutions.drawing_utils

fingertips = [4, 8, 12, 16, 20]
tip_trails = {}
for fingertip in fingertips:
    tip_trails[fingertip] = []

pTime = 0
frames = 0
tick = 0.3
fps = "0 FPS"

while True:
    success, image = cap.read()
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, c = image.shape

    results = hands.process(imgRGB)
    # first line only required for several hands
    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            # for idx, landmark in enumerate(hand_landmark.landmark):
            # for idx in fingertips:
            #     landmark = hand_landmark.landmark[idx]
            #     cx, cy = int(w*landmark.x), int(h*landmark.y)
            #     cv2.circle(image, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

            for fingertip in [8]:
                tip_lm = hand_landmark.landmark[fingertip]
                cx, cy = int(w * tip_lm.x), int(h * tip_lm.y)
                tip_trails[fingertip].append(
                    {"pos": (cx, cy), "lifetime": 90})
                for x in tip_trails[fingertip]:
                    if x["lifetime"] == 0:
                        tip_trails[fingertip].remove(x)
                    else:
                        cv2.circle(image, x["pos"], 5,
                                   (255, 0, 255), cv2.FILLED)
                        x["lifetime"] -= 1

            mpDraw.draw_landmarks(image, hand_landmark,
                                  mpHands.HAND_CONNECTIONS)

    frames += 1
    if time.time_ns() - pTime > tick * 10**9:
        fps = str(int(frames / tick)) + " FPS"
        pTime = time.time_ns()
        frames = 0
    cv2.putText(
        image,
        fps,
        (10, 30),
        cv2.FONT_HERSHEY_PLAIN,
        1.5,
        (0, 0, 0),
        1
    )

    cv2.imshow("Image", image)
    cv2.waitKey(1)
