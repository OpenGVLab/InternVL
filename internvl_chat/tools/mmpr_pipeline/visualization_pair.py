import io
import re
import os
import sys
import json
import base64
import logging
import argparse
import gradio as gr

from collections import defaultdict
from PIL import Image, ImageDraw
from petrel_client.client import Client

client = Client()

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
)

IMAGE_PLACEHOLDER = '<image>'
BOX_SCALE = 999
REF_START_TAG = '<ref>'
REF_END_TAG = '</ref>'
BOX_START_TAG = '<box>'
BOX_END_TAG = '</box>'
REL_START_TAG = '<pred>'
REL_END_TAG = '</pred>'

colors = [(255, 6, 27), (94, 163, 69), (50, 103, 185), (255, 184, 44), (244, 114, 54), (120, 121, 180), (249, 30, 136), (60, 228, 208), (91, 36, 197), (0, 85, 24)]

class Dataset:
    def __init__(self, meta):
        self.image_path = meta['root']
        self.data_path = meta['annotation']

        if 's3://' in self.data_path:
            self.lines = io.BytesIO(client.get(self.data_path)).readlines()
        else:
            with open(self.data_path) as file:
                self.lines = file.readlines()

    def __getitem__(self, index):
        item = self.lines[index]
        item = json.loads(item)
        if 'image' in item:
            item['image'] = os.path.join(self.image_path, item['image'])
            item['image'] = item['image'].replace('wenhaitmp:s3://internvl/', 'langchao:s3://internvl2/')
            item['image'] = load_image(item['image'])
        else:
            item['image'] = None
        return item.copy()

    def __len__(self):
        return len(self.lines)

def load_image(image_file):
    if 's3://' in image_file:
        image_file = client.get(image_file)
        image_file = io.BytesIO(image_file)
    image = Image.open(image_file).convert('RGB')
    return image

def image_to_mdstring(image):
    if isinstance(image, str):
        image = load_image(image)

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"![image](data:image/png;base64,{img_str})"

def draw_box(image, bbox, bbox2category, bbox2cid, width=8, alpha=32, square_pad=False):
    # bbox2cid is used to ensure that the specific box is always drawed with the same color
    x1, y1, x2, y2 = bbox
    category = bbox2category[json.dumps(bbox)]

    if square_pad:
        bbox = bbox.copy()
        bbox = [
            bbox[0] / BOX_SCALE * max(image.height, image.width),
            bbox[1] / BOX_SCALE * max(image.height, image.width),
            bbox[2] / BOX_SCALE * max(image.height, image.width),
            bbox[3] / BOX_SCALE * max(image.height, image.width),
        ]

        if image.height == image.width:
            pass
        elif image.height < image.width:
            delta = (image.width - image.height) // 2
            bbox[1] -= delta
            bbox[3] -= delta
        else:
            delta = (image.height - image.width) // 2
            bbox[0] -= delta
            bbox[2] -= delta

        for i in range(len(bbox)):
            if bbox[i] < 0:
                bbox[i] = 0

        bbox = tuple(bbox)

    else:
        bbox = (
            x1 / BOX_SCALE * image.width,
            y1 / BOX_SCALE * image.height,
            x2 / BOX_SCALE * image.width,
            y2 / BOX_SCALE * image.height,
        )

    if bbox not in bbox2cid:
        bbox2cid[bbox] = len(bbox2cid) % len(colors)

    # draw box
    ImageDraw.Draw(image).rectangle(bbox, outline=colors[bbox2cid[bbox]], width=width)

    # fill box
    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    rectangle_position = bbox
    rectangle_color = (*colors[bbox2cid[bbox]], alpha)
    draw.rectangle(rectangle_position, fill=rectangle_color)
    image = Image.alpha_composite(image.convert('RGBA'), overlay)

    return image

def extract_objects(
    grounded_caption: str,
    grounded_pattern: str = r'<.*?>.*?<.*?>',
):
    objects = defaultdict(list)
    relations = defaultdict(list)
    res = re.findall(grounded_pattern, grounded_caption)

    clean_caption = grounded_caption
    clean_caption = clean_caption.replace(REF_START_TAG, '').replace(REF_END_TAG, '')
    clean_caption = clean_caption.replace(REL_START_TAG, '').replace(REL_END_TAG, '')

    # last_item = None
    last_item = 'None'
    objects[last_item] = []

    last_tag = 'None'
    last_tag_value = 'None'
    for item in res:
        clean_item = re.sub(r'<.*?>', '', item)

        if item.startswith(BOX_START_TAG):
            clean_caption = clean_caption.replace(item, '')
            clean_item = json.loads(clean_item)
            if last_tag in [REF_START_TAG, 'None']:
                objects[last_tag_value].extend(clean_item)
            elif last_tag == REL_START_TAG:
                relations[last_tag_value].append(clean_item)
            else:
                raise NotImplementedError(grounded_caption)
        else:
            last_tag = REF_START_TAG if item.startswith(REF_START_TAG) else REL_START_TAG
            last_tag_value = clean_item

    bbox2category = defaultdict(list)
    for k, v in objects.items():
        for bbox in v:
            bbox2category[json.dumps(bbox)].append(k)

    return objects, relations, bbox2category, clean_caption

def visualize_objects(md_str, image):
    images = []
    objects, relations, bbox2category, clean_caption = extract_objects(md_str)

    # visualize objects
    bbox2cid = {}
    for obj_name, bbox_list in objects.items():
        for bbox in bbox_list:
            image_to_draw = image.copy()
            image_to_draw = draw_box(image=image_to_draw, bbox=bbox, bbox2category=bbox2category, bbox2cid=bbox2cid)
            images.append(image_to_draw)

    # extract scene graph
    scene_graph = []
    for rel_name, bbox_list in relations.items():
        assert len(bbox_list) % 2 == 0
        for i in range(0, len(bbox_list), 2):
            subject_bboxes = bbox_list[i]
            object_bboxes = bbox_list[i+1]

            if len(subject_bboxes) == 1:
                subject_bboxes = subject_bboxes * len(object_bboxes)

            if len(object_bboxes) == 1:
                object_bboxes = object_bboxes * len(subject_bboxes)

            assert len(subject_bboxes) == len(object_bboxes)
            for subj_bbox, obj_bbox in zip(subject_bboxes, object_bboxes):
                subj = bbox2category[json.dumps(subj_bbox)]
                obj = bbox2category[json.dumps(obj_bbox)]
                scene_graph.append((subj, subj_bbox, obj, obj_bbox, rel_name))

    # visualize nodes in the scene graph
    for rel_name, bbox_list in relations.items():
        for bboxes in bbox_list:
            for bbox in bboxes:
                image_to_draw = image.copy()
                image_to_draw = draw_box(image=image_to_draw, bbox=bbox, bbox2category=bbox2category, bbox2cid=bbox2cid)
                images.append(image_to_draw)

    # visualize edges in the scene graph
    for rel_name, bbox_list in relations.items():
        relation_cnt = 0
        for bboxes in bbox_list:
            if relation_cnt % 2 == 0:
                image_to_draw = image.copy()

            for bbox in bboxes:
                image_to_draw = draw_box(image=image_to_draw, bbox=bbox, bbox2category=bbox2category, bbox2cid=bbox2cid, alpha=64)

            relation_cnt += 1
            if relation_cnt % 2 == 0:
                images.append(image_to_draw)

        if relation_cnt % 2 != 0:
            print(f"Format Warning: {rel_name}, {relation_cnt}")

    return images

def process_item(context, image, title):
    md_str = []
    md_str.append(f"### {title}")
    md_str.append(context.replace('\n', '\n\n'))
    md_str = '\n\n'.join(md_str)

    if image is not None:
        md_str = md_str.replace(IMAGE_PLACEHOLDER, image_to_mdstring(image.copy()))
        images_with_bbox = visualize_objects(md_str, image.copy())
        for idx, image_with_bbox in enumerate(images_with_bbox):
            md_str = f'{md_str}\n\n### Image with region {idx}\n\n{image_to_mdstring(image_with_bbox.copy())}'

    return md_str.replace('\\', '\\\\').replace('$', '\\$').replace('<', '\\<').replace('>', '\\>')

def meta2str(meta):
    if 'claims' in meta:
        claims = '\n\n'.join(meta['claims'])
        claims_converted = '\n\n'.join(meta['claims_converted'])
        results = '\n\n'.join(meta['results'])

        return (
            f'### Claims\n\n'
            f'{claims}\n\n'
            f'### Claims (converted)\n\n'
            f'{claims_converted}\n\n'
            f'### Results\n\n'
            f'{results}\n\n'
        )
    else:
        return ''


def meta2str_v2(meta):
    objects = '\n\n'.join(meta['objects'])
    results = '\n\n'.join(meta['results'])
    hallucinations = '\n\n'.join(meta['hallucinations'])

    return (
        f'### Objects\n\n'
        f'{objects}\n\n'
        f'### Results\n\n'
        f'{results}\n\n'
        f'### Hallucinations\n\n'
        f'{hallucinations}\n\n'
    )

def gradio_app_vis(args):
    with open(args.meta_path) as file:
        meta_info = json.load(file)

    def load_and_collate_annotations(user_state, ann_filename):
        dataset = Dataset(user_state['meta_info'][ann_filename])
        return dataset

    def when_btn_next_click(user_state, ann_filename, ann_id, q_ann, c_ann, r_ann):
        ann_id = int(ann_id) + 1
        item = user_state[ann_filename][ann_id]

        if '<image>' not in item['question'] and item['image'] is not None:
            item['question'] = f"<image>\n{item['question']}"

        is_tie = item.get('is_tie', False)
        q_ann = process_item(item['question'], item['image'], f'Question ({is_tie=})')
        if 'answer_gt' in item:
            q_ann += '\n\n' + process_item(item['answer_gt'], item['image'], 'Answer')
        c_ann = process_item(item['chosen'], item['image'], 'Chosen')
        r_ann = process_item(item['rejected'], item['image'], 'Rejected')

        if 'meta' in item:
            c_meta = process_item(meta2str_v2(item['meta']), None, 'Meta')
            c_ann = f'{c_ann}\n\n{c_meta}'

        if 'chosen_meta' in item:
            c_meta = process_item(meta2str(item['chosen_meta']), None, 'Meta')
            c_ann = f'{c_ann}\n\n{c_meta}'

        if 'rejected_meta' in item:
            r_meta = process_item(meta2str(item['rejected_meta']), None, 'Meta')
            r_ann = f'{r_ann}\n\n{r_meta}'

        return ann_filename, ann_id, q_ann, c_ann, r_ann

    def when_btn_reset_click(user_state, ann_filename, ann_id, q_ann, c_ann, r_ann):
        return when_btn_next_click(user_state, ann_filename, -1, q_ann, c_ann, r_ann)

    def when_ann_filename_change(user_state, ann_filename, ann_id, q_ann, c_ann, r_ann):
        obj = user_state.get(ann_filename, None) 
        if obj is None:
            obj = load_and_collate_annotations(user_state, ann_filename)
            user_state[ann_filename] = obj

        return when_btn_next_click(user_state, ann_filename, -1, q_ann, c_ann, r_ann)

    with gr.Blocks() as app:
        ann_filename = gr.Radio(sorted(list(meta_info.keys())), value=None)
        with gr.Row():
            filepath = gr.Text(args.meta_path, interactive=False)
            ann_id = gr.Number(0)
            btn_next = gr.Button("Next")
            btn_reset = gr.Button("Reset")

        with gr.Row():
            question = gr.Markdown()

        with gr.Row():
            chosen = gr.Markdown()
            rejected = gr.Markdown()

        user_state = gr.State({'meta_info': meta_info})
        all_components = [ann_filename, ann_id, question, chosen, rejected]
        ann_filename.change(when_ann_filename_change, [user_state] + all_components, all_components)
        btn_reset.click(when_btn_reset_click, [user_state] + all_components, all_components)
        btn_next.click(when_btn_next_click, [user_state] + all_components, all_components)

    server_port = 10011
    for i in range(10011, 10100):
        cmd = f'netstat -aon|grep {i}'
        with os.popen(cmd, 'r') as file:
            if '' == file.read():
                server_port = i
                break
    app.launch(share=True, server_port=server_port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta-path', type=str, default='dpo_data/OpenGVLab_InternVL2-8B_clean/meta.json')
    args = parser.parse_args()
    gradio_app_vis(args)
