import base64
import io
import json

from PIL import Image

# Assuming base64_str is the string value without 'data:image/jpeg;base64,'

jsonl_path = './test_texts.jsonl'
f = open(jsonl_path, 'r')
lines = f.readlines()

filename2captions = {}
for line in lines:
    line = json.loads(line)
    imageid = line['image_ids'][0]
    filename = str(imageid) + '.jpg'
    caption = line['text']  # .encode("unicode_escape").decode()
    if filename not in filename2captions:
        filename2captions[filename] = []
    filename2captions[filename].append(caption)
f.close()

images = []
annotations = []
ann_id = 0

# assert each one in filename2captions has 5 captions
for k, v in filename2captions.items():
    image_id = int(k.split('.')[0])
    file_name = k
    file_name = 'images/' + file_name
    images.append({
        'id': image_id,
        'file_name': file_name
    })
    for item in v:
        annotations.append({
            'id': ann_id,
            'image_id': image_id,
            'caption': item
        })
        ann_id += 1

print(len(images))
print(len(annotations))

out = {
    'images': images,
    'annotations': annotations
}
# write to json
f = open('coco_cn_clip/coco-cn_test.json', 'w')
json.dump(out, f)

f = open('./test_imgs.tsv', 'r')
test_images = [item.replace('\n', '') for item in f.readlines()]
for line in test_images:
    file_id, base64i = line.split('\t')
    filename = file_id + '.jpg'
    img = Image.open(io.BytesIO(base64.decodebytes(bytes(base64i, 'utf-8'))))
    img.save('coco_cn_clip/images/' + filename)
    print(filename, img.size)
    # convert base64i to image

f.close()
