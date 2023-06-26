# yolov5_sort

This is an implementation of a SORT object tracker that utilises a YOLOV5 detector.

## Run with docker
Use this container from Nvidia (https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_22-06.html#rel_22-06) and use this command.
```
docker run -dit --shm-size=8g --gpus all nvcr.io/nvidia/pytorch:22.06-py3
```

Install the following packages:
```
pip install scikit-image filterpy seaborn
pip install ultralytics --no-deps
```

(You will need to install ultralytics with --no-deps here because otherwise it will install opencv-python but opencv is already included in the container.)

## Produce Annotations
You can produce COCO-style bounding box annotations on a dataset of images. 

```
python tools/build_dataset.py --image-dir my_dataset --out my_data.json --classes sugar_beet weed --agnostic-nms
```