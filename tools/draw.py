import cv2

PALETTE = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in PALETTE]
    return tuple(color)

def draw_box(img, box, id, cat, names=None, offset=(0, 0)):
    x1, y1, x2, y2 = [int(i) for i in box]
    x1 += offset[0]
    x2 += offset[0]
    y1 += offset[1]
    y2 += offset[1]

    color = compute_color_for_labels(id)

    label = f'{names[cat]} | {id}'
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(
        img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
    cv2.putText(img, label, (x1, y1 +
                                t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img

def draw_boxes(model, tracker, image_dir, class_names, range=range(0, 100)):
    for img_file in sorted(os.listdir(image_dir))[range]:
        img = cv2.imread(os.path.join(image_dir, img_file))
        detect(model, tracker, img)

        if draw:
            cv2.imwrite(f'images/{img_file}', im)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='models/best.pt')
    parser.add_argument('--image-dir', type=str, default='images')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
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

        draw_boxes(model, sort_tracker, image_dir)