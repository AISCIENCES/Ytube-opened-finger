import cv2
import mediapipe as mp

cap = cv2.VideoCapture("fingers.mp4")
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    multiLandMarks = results.multi_hand_landmarks
    if multiLandMarks:
        handPoints = []
        for handLms in multiLandMarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            for idx, lm in enumerate(handLms.landmark):
                # print(idx, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(cx, cy)
                handPoints.append((cx, cy))

        for point in handPoints:
            cv2.circle(img, point, 10, (0,0,255), cv2.FILLED)


        fingersState = [0,0,0,0,0]

        if handPoints[8][1] < handPoints[6][1]:
            fingersState[0] = 1
        if handPoints[12][1] < handPoints[10][1]:
            fingersState[1] = 1
        if handPoints[16][1] < handPoints[14][1]:
            fingersState[2] = 1
        if handPoints[20][1] < handPoints[18][1]:
            fingersState[3] = 1

        if handPoints[4][0] > handPoints[2][0]:
            fingersState[4] = 1

        print(fingersState)



    cv2.imshow("Fingers", img)
    cv2.waitKey(1)
