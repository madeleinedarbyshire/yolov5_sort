import argparse
import cv2
import numpy as np
import os
import torch

from draw import draw_box
from sort import Sort

def initialize_detector(weights, device, agnostic_nms, conf_thres, iou_thres):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, device=device)
    model.conf = conf_thres  # NMS confidence threshold
    model.iou = iou_thres  # NMS IoU threshold
    model.agnostic = agnostic_nms # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 100  # maximum number of detections per image
    return model

def detect(model, tracker, img, img_size=640, class_names=None, draw=False):
    dets = model(img, size=img_size)
    dets_to_sort = np.empty((0,6))

    for det in dets.xyxy[0].cpu().detach().numpy():
        dets_to_sort = np.vstack((dets_to_sort, np.array(det)))

    tracked_dets, _ = tracker.update(dets_to_sort)

    instances = []
    for x1, y1, x2, y2, cat, _, _, _, id in tracked_dets:
        if draw:
            img = draw_box(img, box, id, cat, names=class_names)
        w, h = abs(x2-x1), abs(y2-y1)
        instances.append({'bbox': [x1, y1, w, h],
                          'area': w*h,
                          'category_id': int(cat),
                          'instance_id': int(id)})
    if draw:
        cv2.imwrite(f'images/{img_file}', im)

    return instances
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='models/best.pt')
    parser.add_argument('--source', type=str, default='../sugarbeet2023/070623/2023-06-07-14-51-42')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', type=int, default=2)
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--sort-max-age', type=int, default=20,help='keep track of object even if object is occluded or not detected in n frames')
    parser.add_argument('--sort-min-hits', type=int, default=3, help='start tracking only after n number of objects detected')
    parser.add_argument('--sort-iou-thresh', type=float, default=0.3,help='intersection-over-union threshold between two frames for association')
    
    args = parser.parse_args()

    with torch.no_grad():
        model = initialize_detector(args.weights, args.device, args.agnostic_nms, args.conf_thres, args.iou_thres)
        sort_tracker = Sort(max_age=args.sort_max_age,
                            min_hits=args.sort_min_hits,
                            iou_threshold=args.sort_iou_thresh)
        detect(model, sort_tracker, args.source, args.img_size)