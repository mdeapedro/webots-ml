from controller import Robot, Keyboard, Camera
import numpy as np
import cv2
import os
import time

TIME_STEP = 64
DATASET_DIR = 'dataset/images'
os.makedirs(DATASET_DIR, exist_ok=True)

robot = Robot()

keyboard = Keyboard()
keyboard.enable(TIME_STEP)

camera = robot.getDevice('camera')
camera.enable(TIME_STEP)

fl_wheel = robot.getDevice('front left wheel')
fr_wheel = robot.getDevice('front right wheel')
bl_wheel = robot.getDevice('back left wheel')
br_wheel = robot.getDevice('back right wheel')

fl_wheel.setPosition(float('inf'))
fr_wheel.setPosition(float('inf'))
bl_wheel.setPosition(float('inf'))
br_wheel.setPosition(float('inf'))

fl_wheel.setVelocity(0)
fr_wheel.setVelocity(0)
bl_wheel.setVelocity(0)
br_wheel.setVelocity(0)

MAX_SPEED = 5.0
last_save_time = 0
save_interval = 0.5  # Salvar a cada 0.5s quando movendo

while robot.step(TIME_STEP) != -1:
    key = keyboard.getKey()
    left_speed = 0
    right_speed = 0

    if key == ord('W'):
        left_speed = MAX_SPEED
        right_speed = MAX_SPEED
    elif key == ord('S'):
        left_speed = -MAX_SPEED / 2
        right_speed = -MAX_SPEED / 2
    elif key == ord('A'):
        left_speed = -MAX_SPEED
        right_speed = MAX_SPEED
    elif key == ord('D'):
        left_speed = MAX_SPEED
        right_speed = -MAX_SPEED

    fl_wheel.setVelocity(left_speed)
    bl_wheel.setVelocity(left_speed)
    fr_wheel.setVelocity(right_speed)
    br_wheel.setVelocity(right_speed)

    if key != -1 and time.time() - last_save_time > save_interval:
        image_data = np.frombuffer(camera.getImage(), np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
        image = cv2.cvtColor(image_data, cv2.COLOR_BGRA2BGR)
        filename = f'{DATASET_DIR}/image_{int(time.time())}.png'
        cv2.imwrite(filename, image)
        last_save_time = time.time()
