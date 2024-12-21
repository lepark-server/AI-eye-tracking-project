# Face + Iris Landmarks Real-time Detection + Head pose estimation using face landmarks in C++ (OpenCV + Tensorflow Lite)
<div align="center">
   <img src=https://github.com/lepark-server/AI-eye-tracking-project/blob/main/demo/Demo.png?raw=true"
   width="300">
</div>

## (Note: This guide is for Ubuntu, but the code should work fine on other OS, too)

This project runs on Mediapipe TFLite models without using Mediapipe framework. It can run at **30+ FPS** on **CPU**. 
I perform the test on an I5 10th and the app takes about 5% CPU while running.
For more information:
* Face detection: https://google.github.io/mediapipe/solutions/face_detection.html
* Face landmarks: https://google.github.io/mediapipe/solutions/face_mesh.html
* Iris landmarks: https://google.github.io/mediapipe/solutions/iris.html

## :warning: Why not using GPU ?
Because Tensorflow Lite only supports GPU delegate for Android and IOS.
For more information: https://www.tensorflow.org/lite/performance/gpu

## :computer: Requirements:

### Hardware: Ubuntu 22.04

### CMake >= 3.16
You can follow instructions at https://www.40tude.fr/compile-cpp-code-with-vscode-cmake-nmake/

### OpenCV (for Demo)
#### Install Prebuilt OpenCV

##### Step 1: Update and Upgrade System
```bash
sudo apt update
sudo apt upgrade
```
##### Step 2: Install OpenCV Libraries
```bash
sudo apt install libopencv-dev
pkg-config --modversion opencv4
```
##### Step 3: Verify Installation
```bash
pkg-config --modversion opencv4
```
### Tensorflow Lite
<details>
  <summary>How to use pre-built library</summary>

1. Download and extract tensorflowlite.zip from https://github.com/shigure3011/mediapipe_face_iris_cpp/releases
2. Change `TFLite_PATH` in CMakeLists.txt
3. Add `TFLite_LIBS` to PATH 

</details>

## :key: How to use:
1. Clone this repo and go to FaceMeshCpp folder
2. Run `cmake -S . -B build`
3. Run `cmake --build build --config Release --target FaceMeshCpp`
4. Now it will build an `.exe` at `~/build/Release`. Make sure to copy `model` folder to `~/build/Release/` before running.
