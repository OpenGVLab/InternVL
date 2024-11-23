import argparse
import json
import re

import torch
from torchvision.ops.boxes import box_area


def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    intersection_x1 = max(x1, x3)
    intersection_y1 = max(y1, y3)
    intersection_x2 = min(x2, x4)
    intersection_y2 = min(y2, y4)

    intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * max(
        0, intersection_y2 - intersection_y1 + 1
    )

    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x4 - x3 + 1) * (y4 - y3 + 1)

    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area

    return iou


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def transform_bbox(bbox, image_size):
    x1, y1, x2, y2 = bbox
    W, H = image_size
    x1 = min(max(x1 / 1000 * W, 0), W)
    x2 = min(max(x2 / 1000 * W, 0), W)
    y1 = min(max(y1 / 1000 * H, 0), H)
    y2 = min(max(y2 / 1000 * H, 0), H)

    return [x1, y1, x2, y2]


def evaluation_metrics(outputs):
    correct = 0
    incorrect = 0
    pattern = r'\[*\[.*?,.*?,.*?,.*?\]\]*'
    # pattern = r'\[*\[(.*?),(.*?),(.*?),(.*?)\]\]*'
    # print(outputs)
    for output in outputs:
        bbox = output['gt_answers']
        image_size = output['image_size']
        pred = output['answer']
        # 查找所有匹配
        matches = re.findall(pattern, pred)
        if len(matches) > 1:
            print('大于一个匹配')
            print(matches)
        if len(matches) == 0:
            incorrect = incorrect + 1
        else:
            try:
                pred_bbox = json.loads(matches[0])
                pred_bbox = transform_bbox(pred_bbox[0], image_size)
                iou_score = calculate_iou(pred_bbox, bbox)
                if iou_score > 0.5:
                    correct = correct + 1
                else:
                    incorrect = incorrect + 1
            except Exception as e:
                print(e)
                print(output)
                incorrect = incorrect + 1

        # else:
        #     continue
    print('correct:', correct)
    print('incorrect:', incorrect)
    print('Total:', correct + incorrect)
    print('Acc@0.5:', (correct / (correct + incorrect)))

    return {
        'correct:': correct,
        'incorrect:': incorrect,
        'Total:': correct + incorrect,
        'Acc@0.5:': correct / (correct + incorrect)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default='')
    args = parser.parse_args()
    with open(args.output_file, 'r') as f:
        data = json.load(f)
    if 'outputs' in data:
        data = data['outputs']
    outputs = data
    results = evaluation_metrics(outputs)
    results_file = args.output_file
    with open(results_file, 'w') as f:
        json.dump({
            'results': results,
            'outputs': outputs
        }, f, indent=4)
