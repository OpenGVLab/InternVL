# convert flickr30k_test_karpathy.txt to flickr30k_test_karpathy.json
import json

from pycocotools.coco import COCO


def convert_to_json(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()[1:]

    assert len(lines) == 5000

    data = {}
    data['images'] = []
    data['annotations'] = []
    id = 0
    temp = []
    for line in lines:
        line = line.strip()
        items = line.split(',')
        filename = items[0]
        caption = items[1]
        path = 'Images/' + filename
        image_id = int(filename.split('.')[0])
        print(path, image_id, caption)
        if image_id not in temp:
            data['images'].append({'id': image_id, 'file_name': path})
            temp.append(image_id)
        data['annotations'].append({'image_id': image_id, 'caption': caption, 'id': id})
        id += 1
    assert len(data['images']) == 1000
    assert len(data['annotations']) == 5000

    with open(output_file, 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    convert_to_json('flickr30k_test_karpathy.txt', 'flickr30k_test_karpathy.json')
    coco = COCO('flickr30k_test_karpathy.json')
    print(coco.dataset)
