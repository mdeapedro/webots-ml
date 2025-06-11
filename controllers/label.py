import os
import cv2
import numpy as np

IMAGES_DIR = 'dataset/images'
LABELS_DIR = 'dataset/labels'
os.makedirs(LABELS_DIR, exist_ok=True)

# Faixas HSV (ajuste se necessário baseado em iluminação do sim)
# Objective (amarelo/laranja): hue ~15-30
lower_orange = np.array([10, 80, 80])
upper_orange = np.array([30, 255, 255])

# Red (caixas): hue 0-10 e 170-180
lower_red1 = np.array([0, 80, 80])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 80, 80])
upper_red2 = np.array([180, 255, 255])

# Blue (bolas): hue ~110-130
lower_blue = np.array([100, 80, 80])
upper_blue = np.array([130, 255, 255])

MIN_AREA = 50  # Área mínima para considerar detecção

for filename in os.listdir(IMAGES_DIR):
    if filename.endswith('.png'):
        img_path = os.path.join(IMAGES_DIR, filename)
        img = cv2.imread(img_path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        height, width = img.shape[:2]

        labels = []

        # Detect objective (class 0)
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        contours, _ = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > MIN_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                cx = (x + w / 2) / width
                cy = (y + h / 2) / height
                nw = w / width
                nh = h / height
                labels.append(f'0 {cx} {cy} {nw} {nh}')

        # Detect boxes (red, class 1)
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
        contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > MIN_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                cx = (x + w / 2) / width
                cy = (y + h / 2) / height
                nw = w / width
                nh = h / height
                labels.append(f'1 {cx} {cy} {nw} {nh}')

        # Detect balls (blue, class 2)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > MIN_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                cx = (x + w / 2) / width
                cy = (y + h / 2) / height
                nw = w / width
                nh = h / height
                labels.append(f'2 {cx} {cy} {nw} {nh}')

        if labels:
            label_path = os.path.join(LABELS_DIR, filename.replace('.png', '.txt'))
            with open(label_path, 'w') as f:
                f.write('\n'.join(labels))
