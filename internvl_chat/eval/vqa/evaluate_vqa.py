import argparse
import itertools
import json
import os
import random
import subprocess
import time
from functools import partial
from typing import Optional

import torch
import torchvision.transforms as T
from husky.model.internvl_hf_stage3_v13 import InternChatModel
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import LlamaTokenizer
from vqa import VQA
from vqa_eval import VQAEval

ds_collections = {
    'vqav2_val': {
        'train': 'data/vqav2/vqav2_train.jsonl',
        'test': 'data/vqav2/vqav2_val.jsonl',
        'question': 'data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json',
        'annotation': 'data/vqav2/v2_mscoco_val2014_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vqav2_testdev': {
        'train': 'data/vqav2/vqav2_train.jsonl',
        'test': 'data/vqav2/vqav2_testdev.jsonl',
        'metric': None,
        'max_new_tokens': 10,
    },
    'okvqa_val': {
        'train': 'data/okvqa/okvqa_train.jsonl',
        'test': 'data/okvqa/okvqa_val.jsonl',
        'question': 'data/okvqa/OpenEnded_mscoco_val2014_questions.json',
        'annotation': 'data/okvqa/mscoco_val2014_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'textvqa_val': {
        'train': 'data/textvqa/textvqa_train.jsonl',
        'test': 'data/textvqa/textvqa_val.jsonl',
        'question': 'data/textvqa/textvqa_val_questions.json',
        'annotation': 'data/textvqa/textvqa_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'textvqa_val_ocr': {
        'train': 'data/textvqa/textvqa_train.jsonl',
        'test': 'data/textvqa/textvqa_val_llava.jsonl',
        'question': 'data/textvqa/textvqa_val_questions.json',
        'annotation': 'data/textvqa/textvqa_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vizwiz_val': {
        'train': 'data/vizwiz/vizwiz_train.jsonl',
        'test': 'data/vizwiz/vizwiz_val.jsonl',
        'question': 'data/vizwiz/vizwiz_val_questions.json',
        'annotation': 'data/vizwiz/vizwiz_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vizwiz_test': {
        'train': 'data/vizwiz/vizwiz_train.jsonl',
        'test': 'data/vizwiz/vizwiz_test.jsonl',
        'metric': None,
        'max_new_tokens': 10,
    },
    'docvqa_val': {
        'train': 'data/docvqa/train.jsonl',
        'test': 'data/docvqa/val.jsonl',
        'annotation': 'data/docvqa/val/val_v1.0.json',
        'metric': 'anls',
        'max_new_tokens': 100,
    },
    'docvqa_test': {
        'train': 'data/docvqa/train.jsonl',
        'test': 'data/docvqa/test.jsonl',
        'metric': None,
        'max_new_tokens': 100,
    },
    'chartqa_test_human': {
        'train': 'data/chartqa/train_human.jsonl',
        'test': 'data/chartqa/test_human.jsonl',
        'metric': 'relaxed_accuracy',
        'max_new_tokens': 100,
    },
    'chartqa_test_augmented': {
        'train': 'data/chartqa/train_augmented.jsonl',
        'test': 'data/chartqa/test_augmented.jsonl',
        'metric': 'relaxed_accuracy',
        'max_new_tokens': 100,
    },
    'gqa_testdev': {
        'train': 'data/gqa/train.jsonl',
        'test': 'data/gqa/test_balanced.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    },
    'gqa_testdev_llava': {
        'train': 'data/gqa/train.jsonl',
        'test': 'data/gqa/llava_gqa_testdev_balanced_qwen_format.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    },
    'ocrvqa_val': {
        'train': 'data/ocrvqa/ocrvqa_train.jsonl',
        'test': 'data/ocrvqa/ocrvqa_val.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 100,
    },
    'ocrvqa_test': {
        'train': 'data/ocrvqa/ocrvqa_train.jsonl',
        'test': 'data/ocrvqa/ocrvqa_test.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 100,
    },
    'ai2diagram_test': {
        'train': 'data/ai2diagram/train.jsonl',
        'test': 'data/ai2diagram/test.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    }
}


# https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py#L81
def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith('%'):
                # Convert percentages to floats.
                return float(text.rstrip('%')) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float -
                              target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


def evaluate_relaxed_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            relaxed_correctness(elem['answer'].strip(), ann)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)


def evaluate_exact_match_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            (1.0 if
             (elem['answer'].strip().lower() == ann.strip().lower()) else 0.0)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)


def collate_fn(batches, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    questions = [_['question'] for _ in batches]
    question_ids = [_['question_id'] for _ in batches]
    annotations = [_['annotation'] for _ in batches]

    return pixel_values, questions, question_ids, annotations


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class VQADataset(torch.utils.data.Dataset):

    def __init__(self, train, test, prompt, few_shot, input_size=224, pad2square=False):
        self.test = open(test).readlines()
        self.prompt = prompt

        self.few_shot = few_shot
        if few_shot > 0:
            self.train = open(train).readlines()
        if pad2square:
            self.transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Lambda(lambda img: expand2square(img, tuple(int(x * 255) for x in (0.485, 0.456, 0.406)))),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        else:
            self.transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        data = json.loads(self.test[idx].strip())
        image, question, question_id, annotation = data['image'], data[
            'question'], data['question_id'], data.get('answer', None)

        few_shot_prompt = ''
        if self.few_shot > 0:
            few_shot_samples = random.sample(self.train, self.few_shot)
            for sample in few_shot_samples:
                sample = json.loads(sample.strip())
                few_shot_prompt += self.prompt.format(
                    sample['image'],
                    sample['question']) + f" {sample['answer']}"

        image = Image.open(image).convert('RGB')
        pixel_values = self.transform(image).unsqueeze(0)
        question = question + ' ' + self.prompt
        return {
            'question_id': question_id,
            'question': question,
            'pixel_values': pixel_values,
            'annotation': annotation
        }


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def post_process(response):
    response = response.strip().split('.')[0].split(
        ',')[0].split('!')[0].lower()
    if 'is ' in response:
        response = response.split('is ')[1]
    if 'are ' in response:
        response = response.split('are ')[1]
    if 'a ' in response:
        response = response.split('a ')[1]
    if 'an ' in response:
        response = response.split('an ')[1]
    if 'the ' in response:
        response = response.split('the ')[1]
    if ' of' in response:
        response = response.split(' of')[0]
    response = response.strip()
    return response


def evaluate_chat_model():
    base_prompt = 'Answer the question using a single word or phrase.'
    vizwiz_prompt = "When the provided information is insufficient, respond with 'Unanswerable'. "
    ai2d_prompt = 'Please answer the question based on the options mentioned before.'
    model = InternChatModel.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16).cuda().eval()
    tokenizer = LlamaTokenizer.from_pretrained(args.checkpoint)
    tokenizer.add_eos_token = False

    random.seed(args.seed)
    summaries = []
    for ds_name in args.datasets:
        if 'vizwiz' in ds_name:
            input_prompt = vizwiz_prompt + base_prompt
        elif 'ai2d' in ds_name:
            input_prompt = ai2d_prompt
        else:
            input_prompt = base_prompt

        if model.qllama.config.force_image_size is not None:
            image_size = model.qllama.config.force_image_size
        else:
            image_size = model.qllama.config.vision_config.image_size

        dataset = VQADataset(
            train=ds_collections[ds_name]['train'],
            test=ds_collections[ds_name]['test'],
            prompt=input_prompt,
            few_shot=args.few_shot,
            input_size=image_size,
            pad2square=args.pad2square
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(collate_fn, tokenizer=tokenizer),
        )

        outputs = []
        for _, (pixel_values, questions, question_ids, annotations) in tqdm(enumerate(dataloader)):
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            generation_config = dict(
                do_sample=False,
                num_beams=args.num_beams,
                max_new_tokens=ds_collections[ds_name]['max_new_tokens'],
                min_new_tokens=1,
                length_penalty=1,
            )
            pred = model.chat(
                template=args.template,
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=questions[0],
                generation_config=generation_config,
            )
            answers = [pred]
            # answers = [post_process(pred)]

            for question_id, answer, annotation in zip(question_ids, answers, annotations):
                if ds_name in ['vqav2_val', 'vqav2_testdev', 'okvqa_val', 'textvqa_val',
                               'vizwiz_val', 'textvqa_val_ocr']:
                    outputs.append({
                        'question_id': question_id,
                        'answer': answer,
                    })
                elif ds_name in ['docvqa_val', 'infographicsvqa', 'gqa_testdev', 'ocrvqa_val',
                                 'ocrvqa_test', 'gqa_testdev_llava']:
                    outputs.append({
                        'questionId': question_id,
                        'answer': answer,
                        'annotation': annotation,
                    })
                elif ds_name in ['ai2diagram_test']:
                    outputs.append({
                        'image': question_id,
                        'answer': answer,
                        'annotation': annotation,
                    })
                elif ds_name in ['chartqa_test_human', 'chartqa_test_augmented']:
                    outputs.append({
                        'answer': answer,
                        'annotation': annotation,
                    })
                elif ds_name in ['docvqa_test']:
                    outputs.append({
                        'questionId': question_id,
                        'answer': answer,
                    })
                elif ds_name in ['vizwiz_test']:
                    outputs.append({
                        'image': question_id,
                        'answer': answer,
                    })
                else:
                    raise NotImplementedError

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            print(f'Evaluating {ds_name} ...')
            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            results_file = f'{ds_name}_{time_prefix}.json'
            results_file = os.path.join(args.out_dir, results_file)
            json.dump(merged_outputs, open(results_file, 'w'))
            print('Results saved to {}'.format(results_file))

            if ds_collections[ds_name]['metric'] == 'vqa_score':
                vqa = VQA(ds_collections[ds_name]['annotation'],
                          ds_collections[ds_name]['question'])
                results = vqa.loadRes(
                    resFile=results_file,
                    quesFile=ds_collections[ds_name]['question'])
                vqa_scorer = VQAEval(vqa, results, n=2)
                vqa_scorer.evaluate()

                print(ds_name, vqa_scorer.accuracy)
                summaries.append([args.checkpoint, ds_name, vqa_scorer.accuracy])

            elif ds_collections[ds_name]['metric'] == 'anls':
                json.dump(merged_outputs,
                          open(results_file, 'w'),
                          ensure_ascii=False)
                print('python eval/vqa/infographicsvqa_eval.py -g ' +
                      ds_collections[ds_name]['annotation'] + ' -s ' +
                      results_file)
                os.system('python eval/vqa/infographicsvqa_eval.py -g ' +
                          ds_collections[ds_name]['annotation'] + ' -s ' +
                          results_file)
            elif ds_collections[ds_name]['metric'] == 'relaxed_accuracy':
                relaxed_accuracy = evaluate_relaxed_accuracy(merged_outputs)
                print(ds_name, {'relaxed_accuracy': relaxed_accuracy})
                summaries.append([ds_name, {'relaxed_accuracy': relaxed_accuracy}])
            elif ds_collections[ds_name]['metric'] == 'accuracy':
                if 'gqa' in ds_name:
                    dst_file = './data/gqa/testdev_balanced_predictions.json'
                    print('python eval/vqa/convert_gqa_for_eval.py --src ' +
                          results_file + ' --dst ' + dst_file)
                    python_path = '/mnt/afs/user/chenzhe/.conda/envs/husky/bin/python'
                    os.system(python_path + ' eval/vqa/convert_gqa_for_eval.py --src ' +
                              results_file + ' --dst ' + dst_file)
                    command = f'cd ./data/gqa/ && {python_path} eval.py --tier testdev_balanced && cd ../../'
                    print(command)
                    accuracy = subprocess.check_output(command, shell=True, universal_newlines=True)
                else:
                    accuracy = {'accuracy': evaluate_exact_match_accuracy(merged_outputs)}
                print(ds_name, accuracy)
                summaries.append([args.checkpoint, ds_name, accuracy])

        torch.distributed.barrier()

    out_path = '_'.join(args.checkpoint.split('/')[-2:])
    writer = open(os.path.join(args.out_dir, f'{out_path}.txt'), 'a')
    print(f"write results to file {os.path.join(args.out_dir, f'{out_path}.txt')}")
    for summary in summaries:
        print(summary)
        writer.write(f'{summary}\n')
    writer.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str,
                        default='okvqa_val,textvqa_val_ocr,vizwiz_val,ai2diagram_test,gqa_testdev_llava')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--template', type=str, default='vicuna_v1.1')
    parser.add_argument('--out_dir', type=str, default='results')
    parser.add_argument('--few-shot', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pad2square', type=bool, default=False)
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    print('pad2square:', args.pad2square)
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    evaluate_chat_model()
