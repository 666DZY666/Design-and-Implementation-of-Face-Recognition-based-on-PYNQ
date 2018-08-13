# YOLO for PYNQ-Z1 and Intel/Movidius Neural Compute Stick (NCS)

This project is derived from [yoloNCS](https://github.com/gudovskiy/yoloNCS) and is intended to be used on the [PYNQ-Z1](https://store.digilentinc.com/pynq-z1-python-productivity-for-zynq/ "PYNQ-Z1") board.

![alt text](http://www.fpgadeveloper.com/wp-content/uploads/2018/04/pynq-z1-and-movidius-ncs-1-1080x675.jpg "PYNQ-Z1 and Movidius NCS")

## Using this repo on your PYNQ-Z1

To use this code on your PYNQ-Z1, just follow these steps:

1. Install NCSDK in API-mode on your PYNQ-Z1 as explained here: [Setting up the PYNQ-Z1 for the Intel Movidius NCS](http://www.fpgadeveloper.com/2018/04/setting-up-the-pynq-z1-for-the-intel-movidius-neural-compute-stick.html)

2. Clone this repo onto your PYNQ-Z1 in this directory: `/home/xilinx/jupyter_notebooks`

3. Boot the PYNQ-Z1, open Jupyter in a web browser (http://pynq:9090) and open one of the notebooks

## News

* Camera App is working.
* YOLOv1 Tiny is working.

## Protobuf Model files

./prototxt/

## Download Pretrained Caffe Models to ./weights/

* YOLO_tiny: https://drive.google.com/file/d/0Bzy9LxvTYIgKNFEzOEdaZ3U0Nms/view?usp=sharing

## Compilation

* Compile .prototxt and corresponding .caffemodel (with the same name) to get NCS graph file. For example: "mvNCCompile prototxt/yolo_tiny_deploy.prototxt -w weights/yolo_tiny_deploy.caffemodel -s 12"
* The compiled binary file "graph" has to be in main folder after this step.

## Single Image Script

* Run "yolo_example.py" to process a single image. For example: "python3 py_examples/yolo_example.py images/dog.jpg" to get detections as below.

![](/images/yolo_dog.png)

## Camera Input Script

* Run "object_detection_app.py" to process a videos from your camera. For example: "python3 py_examples/object_detection_app.py" to get camera detections as below.
* Modify script arguments if needed.
* Press "q" to exit app.

![](/images/camera.png)
