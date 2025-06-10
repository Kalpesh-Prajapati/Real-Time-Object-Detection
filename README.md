# YOLOv3 Real-Time Object Detection

This project demonstrates real-time object detection using the YOLOv3 (You Only Look Once) deep learning model and OpenCV with webcam input.



## 📁 Files Included

- `object detection live.py` – Main script to run object detection
- `yolov3.cfg` – Configuration file for YOLOv3
- `yolov3.weights` – Pre-trained weights for YOLOv3
- `objects.names` – List of object class names (e.g., person, car, dog, etc.)


### 📄 Folder Structure

Place the following files in one project folder:

```
yolov3-object-detection-live/
├── object detection live.py
├── yolov3.cfg
├── yolov3.weights
├── objects.names
└── README.md
```


## 🚀 Requirements

- Python 3.x
- OpenCV
- NumPy

### 🔧 Install dependencies

```bash
pip install opencv-python numpy
````

## ▶️ How to Run

1. Clone or download this repository.
2. Make sure all files (`.py`, `.cfg`, `.weights`, `.names`) are in the same directory.
3. Run the script:

```bash
python "object detection live.py"
```

4. A webcam window will open showing bounding boxes around detected objects with class names and confidence scores.

## 🎯 Features

* Real-time object detection using YOLOv3
* Labels and confidence scores displayed
* Detects 80 common object types from COCO dataset

## 🔒 License

Pre-trained YOLOv3 weights are licensed by Joseph Redmon (Darknet). Code is provided under the MIT License.

## 🙋‍♂️ Author

Created by Kalpesh Prajapati


1. **Create the project folder**:
   ```bash
   mkdir yolov3-object-detection-live
   cd yolov3-object-detection-live
   ````

2. **Place the following files** into the folder:

   * `object detection live.py`
   * `yolov3.cfg`
   * `yolov3.weights`
   * `objects.names`
   * `README.md`

3. **Initialize Git and push to GitHub**:

   ```bash
   git init
   git add .
   git commit -m "Initial commit: YOLOv3 live object detection"
   git branch -M main
   git remote add origin https://github.com/Kalpesh-Prajapati/yolov3-object-detection-live.git
   git push -u origin main
   ```
