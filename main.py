import cv2
import numpy as np
import pygetwindow as gw
import pyautogui
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import time

# Initialize hand detector and classifier
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300
labels = ["Hello", "I love you", "No", "Okay", "Please", "Thank you", "Yes"]

def find_meet_window(title_keyword="Meet"):
    windows = gw.getWindowsWithTitle(title_keyword)
    for window in windows:
        if window.visible:
            return window
    return None

# Wait for Meet window
print("[INFO] Looking for Meet window...")
time.sleep(3)

meet_window = find_meet_window()

if not meet_window:
    print("❌ Google Meet window not found! Please open it.")
    exit()

print(f"✅ Found window: {meet_window.title}")

# Create a small OpenCV named window and move it aside
cv2.namedWindow("Visual Connect - Meet Capture", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Visual Connect - Meet Capture", 640, 360)  # Small preview
cv2.moveWindow("Visual Connect - Meet Capture", 100, 100)  # Move away from Meet

while True:
    # Get window position and size
    left, top, width, height = meet_window.left, meet_window.top, meet_window.width, meet_window.height

    # Screenshot only the Meet window
    screenshot = pyautogui.screenshot(region=(left, top, width, height))
    img = np.array(screenshot)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    imgOutput = img.copy()

    # Detect hands
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        hImg, wImg, _ = img.shape
        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(wImg, x + w + offset)
        y2 = min(hImg, y + h + offset)

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size != 0:
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Predict
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

            # Show prediction
            cv2.rectangle(imgOutput, (x - offset, y - offset - 70),
                          (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)

            cv2.putText(imgOutput, labels[index], (x, y - 30),
                        cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)

            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (0, 255, 0), 4)

    # Show small output window (not disturbing Meet)
    cv2.imshow('Visual Connect - Meet Capture', imgOutput)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
