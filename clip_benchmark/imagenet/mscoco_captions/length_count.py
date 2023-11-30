import json

import numpy as np

f = open('./coco_test_karpathy.json', 'r')
data = json.load(f)
data = data['annotations']
length = []

for line in data:
    caption = line['caption']
    length.append(len(caption.split()))

length = np.array(length)

print(length.mean(), length.std())
