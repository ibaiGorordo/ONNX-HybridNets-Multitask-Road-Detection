# ONNX-HybridNets-Multitask-Road-Detection
 Python scripts for performing road segemtnation and car detection using the HybridNets multitask model in ONNX.
 
![!HybridNets Road multitask detections](https://github.com/ibaiGorordo/ONNX-HybridNets-Multitask-Road-Detection/blob/main/doc/img/bird_eye_view.png)

# Requirements

 * Check the **requirements.txt** file. Additionally, **pafy** and **youtube-dl** are required for youtube video inference.
 
# Installation
```
pip install -r requirements.txt
pip install pafy youtube_dl>=2021.12.17
```

# ONNX model
The original models were converted to different formats (including .onnx) by [PINTO0309](https://github.com/PINTO0309), download the models from [his repository](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/276_HybridNets) and save them into the **[models](https://github.com/ibaiGorordo/ONNX-HybridNets-Multitask-Road-Detection/tree/main/models)** folder. 

# Original Pytorch model
The Pytorch pretrained model was taken from the [original repository](https://github.com/datvuthanh/HybridNets).
 
# Examples

 * **Image inference**:
 ![!HybridNets Image Road multitask detections](https://github.com/ibaiGorordo/ONNX-HybridNets-Multitask-Road-Detection/blob/main/doc/img/output.jpg)
 
 ```
 image_road_detection.py
 ```

 
