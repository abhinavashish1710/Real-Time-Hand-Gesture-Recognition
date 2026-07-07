import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

tip_ids = [4, 8, 12, 16, 20]

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    gesture = "Unknown"

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            lm = []

            for id, landmark in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                lm.append((int(landmark.x * w), int(landmark.y * h)))

            fingers = []

            # Thumb
            if lm[tip_ids[0]][0] > lm[tip_ids[0] - 1][0]:
                fingers.append(1)
            else:
                fingers.append(0)

            # Other Fingers
            for i in range(1, 5):
                if lm[tip_ids[i]][1] < lm[tip_ids[i] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            if fingers == [0, 0, 0, 0, 0]:
                gesture = "Fist"

            elif fingers == [1, 1, 1, 1, 1]:
                gesture = "Open Palm"

            elif fingers == [0, 1, 0, 0, 0]:
                gesture = "One Finger"

            elif fingers == [0, 1, 1, 0, 0]:
                gesture = "Peace"

            elif fingers == [1, 0, 0, 0, 0]:
                gesture = "Thumbs Up"

            cv2.putText(
                frame,
                gesture,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
