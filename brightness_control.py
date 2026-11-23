
import cv2
import mediapipe as mp
import numpy as np
import screen_brightness_control as sbc
import urllib.request
import os
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Download hand detection model
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading hand model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Initialize MediaPipe
base = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.HandLandmarkerOptions(
    base_options=base,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

# Open webcam
cap = cv2.VideoCapture(0)

last_brightness = -1
last_update = 0

print("Brightness Control Started!")
print("Make a fist and move UP or DOWN.")
print("Press Q to quit.")

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(image)

    mode = "Show your hand"

    if result.hand_landmarks:

        lm = result.hand_landmarks[0]

        # Detect closed fist
        tips = [8, 12, 16, 20]
        joints = [6, 10, 14, 18]

        fingers_up = []

        for tip, joint in zip(tips, joints):
            fingers_up.append(lm[tip].y < lm[joint].y)

        thumb_up = (
            np.hypot(
                lm[4].x - lm[17].x,
                lm[4].y - lm[17].y
            )
            >
            np.hypot(
                lm[3].x - lm[17].x,
                lm[3].y - lm[17].y
            ) * 1.08
        )

        if not any(fingers_up) and not thumb_up:

            mode = "Brightness Control"

            # Palm position
            palm_y = lm[9].y

            # Convert position to brightness
            brightness = int(
                np.interp(palm_y, [0.2, 0.8], [100, 10])
            )

            # Update brightness smoothly
            if (time.time() - last_update > 0.2 and
                abs(brightness - last_brightness) >= 2):

                try:
                    sbc.set_brightness(brightness)
                    last_brightness = brightness
                    last_update = time.time()

                except Exception as e:
                    print("Brightness error:", e)

            cv2.putText(
                frame,
                f"Brightness: {brightness}%",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        else:
            mode = "Make a Closed Fist"

    cv2.putText(
        frame,
        mode,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    cv2.imshow("Hand Gesture Brightness Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close everything
cap.release()
detector.close()
cv2.destroyAllWindows()

print("Brightness Control Stopped!")
