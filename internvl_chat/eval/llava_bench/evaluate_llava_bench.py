import argparse
import json
import os
import random

import torch
from internvl.train.dataset import build_transform
from PIL import Image
from tqdm import tqdm
from transformers import LlamaTokenizer

ds_collections = {
    'llava_bench': {
        'root': 'data/llava-bench-in-the-wild/images',
        'question': 'data/llava-bench-in-the-wild/questions.jsonl',
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
    },
}


class VQADataset(torch.utils.data.Dataset):

    def __init__(self, root, data, prompt, input_size=224, pad2square=False):
        self.root = root
        self.data = open(data).readlines()
        self.prompt = prompt
        self.transform = build_transform(is_train=False, input_size=input_size, pad2square=pad2square)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = json.loads(self.data[idx].strip())
        image, question, question_id, annotation = data['image'], data[
            'text'], data['question_id'], data.get('answer', None)

        image = os.path.join(self.root, image)
        image = Image.open(image).convert('RGB')
        pixel_values = self.transform(image).unsqueeze(0)
        question = question + self.prompt
        return question_id, question, pixel_values, annotation


def evaluate_chat_model():
    random.seed(args.seed)

    for ds_name in args.datasets:
        dataset = VQADataset(
            root=ds_collections[ds_name]['root'],
            data=ds_collections[ds_name]['question'],
            prompt=' Please give a detailed answer.',
            input_size=image_size,
            pad2square=pad2square
        )

        outputs = []
        for _, (question_id, question, pixel_values, annotations) in tqdm(enumerate(dataset)):
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            generation_config = dict(
                num_beams=args.num_beams,
                max_new_tokens=ds_collections[ds_name]['max_new_tokens'],
                min_new_tokens=ds_collections[ds_name]['min_new_tokens'],
                length_penalty=1,
                # repetition_penalty=1.5,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
            )
            pred = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=generation_config,
            )
            outputs.append({
                'question_id': question_id,
                'text': pred,
                'model_id': model_id,
                'metadata': {}
            })

        print(f'Evaluating {ds_name} ...')
        results_file = 'llava_bench_results.jsonl'
        results_file = os.path.join(args.out_dir, results_file)
        writer = open(results_file, 'w')
        for item in outputs:
            writer.write(json.dumps(item) + '\n')
        writer.close()
        print('Results saved to {}'.format(results_file))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='llava_bench')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    tokenizer = LlamaTokenizer.from_pretrained(args.checkpoint)

    if 'qllama' in args.checkpoint.lower():
        from internvl.model.internvl_chat_with_qllama import InternVLChatModel
        model = InternVLChatModel.from_pretrained(
            args.checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).cuda().eval()
        image_size = model.internvl.config.force_image_size or model.config.internvl_config.vision_config.image_size
        pad2square = model.config.pad2square
    else:
        from internvl.model.internvl_chat import InternVLChatModel
        model = InternVLChatModel.from_pretrained(
            args.checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).cuda().eval()
        image_size = model.config.force_image_size or model.config.vision_config.image_size
        pad2square = model.config.pad2square

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    if total_params > 30:
        args.num_beams = 1
        print(f'[test] total_params: {total_params}B, use num_beams: {args.num_beams}')
    else:
        print(f'[test] total_params: {total_params}B')
    print(f'[test] image_size: {image_size}')
    print(f'[test] pad2square: {pad2square}')
    print(f'[test] template: {model.config.template}')

    model_id = '_'.join(args.checkpoint.split('/')[-2:])
    evaluate_chat_model()
