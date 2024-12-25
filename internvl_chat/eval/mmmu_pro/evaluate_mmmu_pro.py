import argparse
import ast
import json
import os

from datasets import load_dataset
from lmdeploy import (ChatTemplateConfig, GenerationConfig,
                      TurbomindEngineConfig, pipeline)
from tqdm import tqdm

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Adjust this if needed
os.environ['HF_HOME'] = './data/MMMU'

argparse = argparse.ArgumentParser()
argparse.add_argument('--model', type=str, default='OpenGVLab/InternVL2-8B')
argparse.add_argument('--mode', type=str, default='direct')
argparse.add_argument('--setting', type=str, default='standard (10 options)')
argparse.add_argument('--tp', type=int, default=1)
args = argparse.parse_args()

MODEL = args.model
MODE = args.mode
SETTING = args.setting
TP = args.tp
MAX_API_RETRY = 5
NUM = 1730

import yaml

with open('eval/mmmu_pro/prompts.yaml', 'r') as file:
    prompt_config = yaml.safe_load(file)[MODE]


def replace_images_tokens(input_string):
    for i in range(1, 8):
        question_text = f'<image {i}>'
        query_text = '<image>'
        if question_text in input_string:
            input_string = input_string.replace(question_text, query_text)
    return input_string


def parse_options(options):
    option_letters = [chr(ord('A') + i) for i in range(len(options))]
    choices_str = '\n'.join([f'{option_letter}. {option}' for option_letter, option in zip(option_letters, options)])
    return choices_str


def construct_prompt(doc):
    question = doc['question']
    parsed_options = parse_options(ast.literal_eval(str(doc['options'])))
    question = f"{question}\n{parsed_options}\n{prompt_config['standard']}"
    return question


def mmmu_doc_to_text(doc):
    question = construct_prompt(doc)
    return replace_images_tokens(question)


def origin_mmmu_doc_to_visual(doc):
    visual = []
    for i in range(1, 8):
        if not doc[f'image_{i}']:
            break
        visual.append(doc[f'image_{i}'])
    return visual


def vision_mmmu_doc_to_visual(doc):
    return [doc['image']]


def process_prompt(data):
    if 'standard' in SETTING:
        prompt = mmmu_doc_to_text(data)
        images = origin_mmmu_doc_to_visual(data)
    elif SETTING == 'vision':
        prompt = prompt_config['vision']
        images = vision_mmmu_doc_to_visual(data)
    return (prompt, images)


def run_and_save(pipe):
    def save_results_to_file(results, output_path):
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for output, data in results:
                data['response'] = output.text
                data = {k: v for k, v in data.items() if not k.startswith('image')}
                json.dump(data, outfile, ensure_ascii=False)
                outfile.write('\n')

    dataset = load_dataset('MMMU/MMMU_Pro', SETTING, split='test')

    # Process and save dataset parts
    def process_and_save_part(part_data, part_name, pipe):
        print(f'Begin processing {part_name}')
        basename = os.path.basename(MODEL)
        os.makedirs('./eval/mmmu_pro/results/', exist_ok=True)
        output_path = f'./eval/mmmu_pro/results/{basename}_{part_name}_{MODE}.jsonl'
        if os.path.exists(output_path):
            print(f'Loaded existing results for {part_name}')
        else:
            responses = []
            for data in tqdm(part_data):
                result = process_prompt(data)
                response = pipe(result, gen_config=gen_config)
                print(response)
                responses.append(response)
            save_results_to_file(zip(responses, part_data), output_path)
        return output_path

    gen_config = GenerationConfig(max_new_tokens=4096, temperature=0.0)

    temp_files = []
    temp_files.append(process_and_save_part(dataset, SETTING, pipe))


if __name__ == '__main__':
    model_name = MODEL.lower().replace('-', '_')
    if 'internvl2_5' in model_name or 'internvl2.5' in model_name:
        pipe = pipeline(MODEL, backend_config=TurbomindEngineConfig(session_len=16384, tp=TP),
                        chat_template_config=ChatTemplateConfig('internvl-internlm2'))
    elif 'internvl2' in model_name:
        pipe = pipeline(MODEL, backend_config=TurbomindEngineConfig(session_len=16384, tp=TP))
    run_and_save(pipe)
