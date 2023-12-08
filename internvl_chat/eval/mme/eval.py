import argparse
import os
import re

import torch
from internvl.model.internvl_chat_with_qllama import (InternVLChatConfig,
                                                      InternVLChatModel)
from internvl.train.dataset import build_transform, expand2square
from PIL import Image
from tqdm import tqdm
from transformers import LlamaTokenizer


def load_image(image_file, input_size=224, pad2square=False):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(is_train=False, input_size=input_size, pad2square=pad2square)
    image = transform(image)
    return image


def post_processing(response):
    response = response.replace('\n', '').replace('不是', 'No').replace('是', 'Yes').replace('否', 'No')
    response = response.lower().replace('true', 'yes').replace('false', 'no')
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    response = re.sub(pattern, '', response)
    return response


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', type=str, default='vicuna_v1.1')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--root', type=str,
                        default='/mnt/petrelfs/wangwenhai/workspace/InternVL-release/internvl_chat/eval/mme/Your_Results')
    parser.add_argument('--beam_num', type=int, default=5)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--sample', type=bool, default=False)
    parser.add_argument('--temperature', type=float, default=1.0)

    args = parser.parse_args()

    prompt = 'Answer the question using a single word or phrase.'
    tokenizer = LlamaTokenizer.from_pretrained(
        args.model_path, add_eos_token=False, legacy=True)

    config = InternVLChatConfig.from_pretrained(args.model_path)
    model = InternVLChatModel.from_pretrained(
        args.model_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, config=config).cuda().eval()
    image_size = model.internvl.config.force_image_size or model.config.internvl_config.vision_config.image_size
    pad2square = model.config.pad2square
    print(f'[MME] force_image_size: {config.internvl_config.force_image_size}')
    print(f'[MME] image_size: {image_size}')
    print(f'[MME] pad2square: {pad2square}')
    print(f'[MME] template: {args.template}')

    output = os.path.basename(args.model_path)
    os.makedirs(output, exist_ok=True)

    for filename in os.listdir(args.root):
        fin = open(os.path.join(args.root, filename), 'r', encoding='utf-8')
        fout = open(os.path.join(output, filename), 'w', encoding='utf-8')
        lines = fin.readlines()
        filename = filename.replace('.txt', '')
        for line in tqdm(lines):
            img, question, gt = line.strip().split('\t')
            question = question + ' ' + prompt
            img_path = os.path.join('images', filename, img)
            assert os.path.exists(img_path), img_path
            pixel_values = load_image(img_path, image_size, pad2square).unsqueeze(0).cuda().to(torch.bfloat16)
            generation_config = dict(
                do_sample=args.sample,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=1.5,
                length_penalty=1.0,
                num_beams=args.beam_num,
                max_new_tokens=20,
                eos_token_id=tokenizer.eos_token_id,
            )
            response = model.chat(
                template=args.template,
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=generation_config,
            )
            response = post_processing(response)
            print(img, question, gt, response, sep='\t', file=fout)
        fin.close()
        fout.close()
