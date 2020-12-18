# Face-masking-with-CV
Final Project on Computer Vision Skoltech course 2020

## Goals
* Attach masks to different parts of face on selfies, videos, in real-time
* Make them move with a person
* Add extra features depending on movements

## CV algorithm

### 1. Keypoints detection
Keypoints detection: dlib pre-trained facial landmark detector (68 points).
![step_1](./images/1.PNG)

### 2. Affine transformation
![step_2](./images/2.PNG)

### 3. Eye blink detection
![step_3](./images/3.PNG)

### 4. Detecting eye circle
![step_4](./images/4.PNG)

### 5. Opened mouth detection
![step_5](./images/5.PNG)

## Using movements

### Eye blink: switch to angel (or devil)
![step_6](./images/6.PNG)

### Open mouth: put protective mask
![step_7](./images/7.PNG)

## Examples
![step_8](./images/8.PNG)

## Want to try?
Check out fast_guide.ipynb

## Tools
* Python 3
* OpenCV
* Keypoint detection: the cascade of regressors using dlib


