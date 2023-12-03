import json

f = open('./coco-cn_test.txt', 'r')
test_filenames = [item.replace('\n', '') for item in f.readlines()]
f.close()

# https://github.com/OFA-Sys/Chinese-CLIP/issues/175
# we should use human-written captions to test retrieval performance
# COCO-CN测评是在测试集，共有1000图像和1053文本，文本为 human-annotated，
# 每个图像与1-2个文本匹配，跟论文中对比的Baseline工作Wukong是一致的
# f = open("./imageid.manually-translated-caption.txt", 'r')
f = open('./imageid.human-written-caption.txt', 'r')
temps = [item.replace('\n', '') for item in f.readlines()]
filename2captions = dict()
for temp in temps:
    temp = temp.split('\t')
    filename = temp[0][:-2]
    caption = temp[1]
    print(filename, caption, len(caption))
    if filename not in test_filenames:
        continue
    if filename not in filename2captions:
        filename2captions[filename] = []
    filename2captions[filename].append(caption)

images = []
annotations = []
ann_id = 0

# assert each one in filename2captions has 5 captions
for k, v in filename2captions.items():
    image_id = int(k.split('_')[-1])
    file_name = k + '.jpg'
    if 'train' in file_name:
        file_name = 'train2014/' + file_name
    else:
        file_name = 'val2014/' + file_name
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
f = open('coco-cn_test.json', 'w')
json.dump(out, f)
