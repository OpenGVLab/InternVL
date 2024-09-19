import argparse
import io
import json
import os
import random

import pandas as pd
from PIL import Image
from tqdm import tqdm

argparse = argparse.ArgumentParser()
argparse.add_argument('input', type=str, default='')
argparse.add_argument('output', type=str, default='')

args = argparse.parse_args()

if not os.path.exists(args.output):
    os.makedirs(args.output)

image_root = os.path.join(args.output, 'images')
if not os.path.exists(image_root):
    os.makedirs(image_root)

prompts = [
    'Please recognize the text in the image.',
    'Please extract the text from the image.',
    'Kindly identify and transcribe the text present in the image.',
    'Could you please perform optical character recognition (OCR) on the image to retrieve the text?',
    'Please use text recognition techniques to decipher the text within the image.',
    'Could you extract any readable text contained in the image?',
    'I need the text within the image recognized and converted into machine-readable format, please.',
    'Please employ OCR technology to recognize and extract the text from the image.',
    'Kindly process the image to identify and retrieve any textual content it contains.',
    'Please analyze the image and retrieve any textual information that is discernible.',
    'Could you transcribe any visible text from the image, please?',
    '请从图像中提取文本',
    '请识别图像中的文本。',
    '能否使用光学字符识别（OCR）技术在图像上提取文本？',
    '请使用文本识别技术解读图像中的文字。',
    '能提取图像中的任何可读文本吗？',
    '请将图像中的文本识别并转换为机器可读格式。',
    '请使用OCR技术识别并提取图像中的文本。',
    '请处理图像以识别并提取其中包含的任何文本内容。',
    '请分析图像并提取其中可以辨认的任何文本信息。',
    '你能够将图像中可见的文本转录出来吗？',
]

cnt = 0
data = []


def process(filename):
    df_parquet = pd.read_parquet(os.path.join(args.input, filename))
    for index, row in tqdm(df_parquet.iterrows()):
        # 在这里对每一行的数据进行处理
        image = row['image']['bytes']
        ground_truth = row['ground_truth']
        ground_truth = json.loads(ground_truth)['gt_parse']['text_sequence']
        image = Image.open(io.BytesIO(image))
        global cnt, data
        image_out_path = os.path.join(args.output, 'images/%08d.jpg' % cnt)
        image.save(image_out_path)
        data_item = {'id': cnt, 'image': 'images/%08d.jpg' % cnt}
        conversations = []
        conversations.append({'from': 'human', 'value': '<image>\n' + random.choice(prompts)})
        conversations.append({'from': 'gpt', 'value': ground_truth})
        data_item['conversations'] = conversations
        data.append(data_item)
        cnt += 1


process('train-00000-of-00084-26dbc51f3d0903b9.parquet')
process('train-00001-of-00084-3efa94914043c815.parquet')
process('train-00002-of-00084-65600b4a95c96e85.parquet')
process('train-00003-of-00084-45e260eca2cd125f.parquet')
process('train-00004-of-00084-b1d684c57dc6e3da.parquet')

writer = open(os.path.join(args.output, 'synthdog_en.jsonl'), 'w')
for item in data:
    writer.write(json.dumps(item) + '\n')
writer.close()

print('done')
