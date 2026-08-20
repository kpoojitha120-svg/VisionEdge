\# VisionEdge



\## AI-Based Real-Time Object Detection System



VisionEdge is a computer vision project that uses the YOLO (You Only Look Once) deep learning model to detect objects in images and through a live webcam. The system performs object detection using CPU-based inference and displays detected object names, confidence scores, and object counts.



\## Features



\* Image-based object detection

\* Real-time webcam object detection

\* YOLO-based detection

\* CPU inference support

\* Confidence score display

\* Total object counting

\* Object-wise counting

\* Configurable model and detection settings



\## Project Structure



```text

VisionEdge/

├── src/

│   ├── config.py

│   ├── main.py

│   └── webcam.py

├── results/

├── bus.jpg

├── yolo11n.pt

├── requirements.txt

├── .gitignore

└── README.md

```



\## Technologies Used



\* Python

\* YOLO

\* Ultralytics

\* OpenCV

\* PyTorch

\* Git

\* GitHub



\## How to Run



\### Image Detection



```powershell

python ".\\src\\main.py"

```



\### Webcam Detection



```powershell

python ".\\src\\webcam.py"

```



Press \*\*Q\*\* inside the camera window to stop webcam detection.



\## Sample Detection



The system can detect objects such as:



\* Bus

\* Person

\* Car

\* Bicycle

\* Other objects supported by the YOLO model



The detection output includes the object name and confidence score. The image detection module also displays the total number of detected objects and the count of each object type.



\## Current Status



The basic VisionEdge detection pipeline is working successfully with YOLO CPU inference and live webcam detection.



