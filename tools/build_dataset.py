import argparse
import cv2
import json
import os
import PIL
import torch
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', 'src'))

from detect import detect, initialize_detector
from sort import Sort

def build(model, tracker, image_dir):
    video_id = 1
    video_name = image_dir.split('/')[-1]
    data = {'videos': {'id': video_id, 'name': video_name},
            'categories': [{'id': 0, 'name': 'sugar_beet'}, {'id': 1, 'name': 'weed'}]}

    images = annotations = []
    image_id = annotation_id = 1

    for img_file in sorted(os.listdir(image_dir)):
        img = cv2.imread(os.path.join(image_dir, img_file))
        img_annotations = detect(model, tracker, img)
        for a in img_annotations:
            a['id'] = annotation_id
            a['video_id'] = 1
            a['image_id'] = image_id
            annotation_id += 1

        h, w, _ = img.shape
        image_data = {'id': image_id,
                      'video_id': video_id,
                      'frame_id': image_id - 1,
                      'file_name': img_file,
                      'width': w,
                      'height': h}

        images.append(image_data)
        annotations.append(img_annotations)
        image_id += 1
        break
    
    data['images'] = images
    data['annotations'] = annotations
    
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='models/best.pt')
    parser.add_argument('--image-dir', type=str, default='images')
    parser.add_argument('--out', type=str, default='dataset.json')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=str, help='List of integers')
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
        dataset = build(model, sort_tracker, args.image_dir)

        with open(args.out, "w") as f:
            json.dump(dataset, f)