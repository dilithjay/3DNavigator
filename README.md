# 3DNavigator

This is the code for a small device that can be used to navigate in a 3D modelling software. The laptop runs an object detection model trained using Tensorflow Object Detection API. This model detects the device and it's motion is used to control the mouse pointer. Other controls such as clicking/panning/rotating/zooming can be done based on the color emitted by the device and the number of fingers the user held up.

v1.py - The code containing the inference and the logic for the controls
led_85.ino - The code that run on the ATtiny85
