import argparse
import cv2
import json
import os
import PIL
import torch
import sys
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', 'src'))

from detect import detect, initialize_detector
from itertools import groupby
from sort import Sort

CLASSES = []

def convert_to_cvat(categories, annotations, dataset_name, total_frames):
    root = ET.Element('annotations')
    tree = ET.ElementTree(root)

    meta = ET.SubElement(root, 'meta')
    name = ET.SubElement(meta, 'name')
    name.text = dataset_name
    
    size = ET.SubElement(meta, 'size')
    size.text = str(total_frames)

    labels = ET.SubElement(root, 'labels')
    for c in CLASSES:
        label = ET.SubElement(labels, 'label')
        label_name = ET.SubElement(label, 'name')
        label_name.text = c

    annotations = sorted(annotations, key=lambda x: x['instance_id'])
    objects = groupby(annotations, key=lambda x: x['instance_id'])

    for track_id, instances in objects:
        instances = list(instances)
        obj = ET.SubElement(root, 'track')
        obj.set('id', str(track_id - 1))
        obj.set('label', categories[instances[0]['category_id']]['name'])
        last_frame = 0
        for inst in sorted(instances, key=lambda x: x['image_id']):
            frame_id = inst['image_id'] - 1

            bbox = ET.SubElement(obj, 'box')
            bbox.set('frame', str(frame_id))
            bbox.set('xtl', str(inst['bbox'][0]))
            bbox.set('ytl', str(inst['bbox'][1]))
            bbox.set('xbr' , str(inst['bbox'][0] + inst['bbox'][2]))
            bbox.set('ybr' , str(inst['bbox'][1] + inst['bbox'][3]))
            bbox.set('outside' , str(0))
            bbox.set('occluded', str(0))
            bbox.set('keyframe', str(1))
            last_frame = frame_id
        if last_frame < total_frames - 1:
            bbox = ET.SubElement(obj, 'box')
            bbox.set('frame', str(last_frame + 1))
            bbox.set('xtl', str(inst['bbox'][0]))
            bbox.set('ytl', str(inst['bbox'][1]))
            bbox.set('xbr' , str(inst['bbox'][0] + inst['bbox'][2]))
            bbox.set('ybr' , str(inst['bbox'][1] + inst['bbox'][3]))
            bbox.set('outside' , str(1))
            bbox.set('occluded', str(0))
            bbox.set('keyframe', str(1))

    return tree

def build_cocovid(model, tracker, image_dir, video_name):
    video_id = 1
    if video_name == None:
        video_name = image_dir.split('/')[-1]
    categories = [{'id': i, 'name': c} for i, c in enumerate(CLASSES)]
    data = {'videos': [{'id': video_id, 'name': video_name}],
            'categories': categories}
    images = []
    annotations = []
    image_id = 1
    annotation_id = 1

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
        annotations = annotations + img_annotations
        image_id += 1
    
    data['images'] = images
    data['annotations'] = annotations
    
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='models/best.pt')
    parser.add_argument('--image-dir', type=str, default='images')
    parser.add_argument('--out', type=str, default='dataset')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--video-name', type=str, help='Video name')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=str, help='List of integers')
    parser.add_argument('--dataset-type', type=str, help='Dataset type')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--sort-max-age', type=int, default=20,help='keep track of object even if object is occluded or not detected in n frames')
    parser.add_argument('--sort-min-hits', type=int, default=3, help='start tracking only after n number of objects detected')
    parser.add_argument('--sort-iou-thresh', type=float, default=0.3,help='intersection-over-union threshold between two frames for association')

    args = parser.parse_args()

    if args.classes:
        CLASSES = args.classes

    with torch.no_grad():
        print('Building dataset for image dir:', args.image_dir)
        model = initialize_detector(args.weights, args.device, args.agnostic_nms, args.conf_thres, args.iou_thres)
        sort_tracker = Sort(max_age=args.sort_max_age,
                            min_hits=args.sort_min_hits,
                            iou_threshold=args.sort_iou_thresh)

        dataset = build_cocovid(model, sort_tracker, args.image_dir, args.video_name)

        if args.dataset_type == 'cvat':
            xml = convert_to_cvat(dataset['categories'], dataset['annotations'], args.out, len(dataset['images']))
            xml.write(f'{args.out}.xml', encoding='utf-8', xml_declaration=True)
        else:  
            with open(f'{args.out}.json', "w") as f:
                json.dump(dataset, f)    