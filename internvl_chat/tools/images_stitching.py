import argparse
import json
import os

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

FOOT = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf', 50)


def custom_image(img_paths, save_path, image_size=448):
    captions = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

    width = image_size * 2
    height = image_size
    # count = 0
    all_images = {}
    for image_id, image_files in tqdm(img_paths.items()):
        all_images[image_id] = dict()
        all_images[image_id]['images_path'] = image_files
        all_images[image_id]['images_size'] = {k: (0, 0) for k in image_files.keys()}
        imgs = {}
        for caption, image_file in image_files.items():
            image_path = os.path.join(args.data_root, image_file.replace('../nuscenes/samples/', '/nuscenes/samples/'))
            img = Image.open(image_path).convert('RGB')
            old_wide, old_height = img.size
            all_images[image_id]['images_size'][caption] = (old_wide, old_height)
            img = img.resize((width, height))

            draw = ImageDraw.Draw(img)
            text = caption
            draw.text((0, 0), text, fill=(255, 0, 255), font=FOOT)
            imgs[caption] = img

        result_width = width * 3
        result_height = height * 2
        result_img = Image.new('RGB', (result_width, result_height))

        imgs = [imgs[caption] for caption in captions]
        for i in range(len(imgs)):
            row = i // 3
            col = i % 3

            left = col * width
            top = row * height
            right = left + width
            bottom = top + height
            result_img.paste(imgs[i], (left, top))

        result_path = os.path.join(save_path, image_id + '.jpg')
        result_img.save(result_path)


def get_images(ann_file):
    with open(ann_file, 'r') as f:  # , \
        train_file = json.load(f)

    images = {}
    for scene_id in train_file.keys():
        scene_data = train_file[scene_id]['key_frames']
        for frame_id in scene_data.keys():
            image_id = scene_id + '_' + frame_id
            if image_id not in images:
                images[image_id] = scene_data[frame_id]['image_paths']
            else:
                print(image_id)

    return images


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='InternVL-Domain-Adaptation-Data/images/drivelm')
    parser.add_argument('--ann-file', type=str, default='path/to/v1_1_val_nus_q_only.json')
    args = parser.parse_args()
    images = get_images(args.ann_file)
    save_path = os.path.join(args.data_root, 'stitch')
    os.makedirs(save_path, exist_ok=True)
    custom_image(img_paths=images, save_path=save_path)
