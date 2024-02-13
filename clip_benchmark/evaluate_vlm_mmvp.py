import argparse
import csv
import os

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor


def benchmark_model(model_name, benchmark_dir, device='cuda'):
    # model_path = 'pretrained/internvl_14b_224px'
    model_path = 'OpenGVLab/InternVL-14B-224px'
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True).cuda().eval()
    preprocess = CLIPImageProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, add_eos_token=True)
    tokenizer.pad_token_id = 0  # set pad_token_id to 0
    image_dir = os.path.join(benchmark_dir, 'MLLM_VLM Images')
    csv_file = os.path.join(benchmark_dir, 'Questions.csv')

    csv_outfile = open('output.csv', 'w', newline='')
    csv_writer = csv.writer(csv_outfile)
    csv_writer.writerow(['qid1', 'qid2', 'pred1', 'pred2', 'gt1', 'gt2', 'q1score', 'q2score'])  # header

    categories = [
        'Orientation and Direction', 'Presence of Specific Features',
        'State and Condition', 'Quantity and Count',
        'Positional and Relational Context', 'Color and Appearance',
        'Structural Characteristics', 'Texts',
        'Viewpoint and Perspective'
    ]

    pair_accuracies = {category: 0 for category in categories}
    num_pairs = 0

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for i, row in tqdm(enumerate(reader)):
            qid1, qtype1, statement1 = row

            # Get next row for the pair
            row = next(reader, None)
            if not row:
                break
            qid2, qtype2, statement2 = row

            qid1, qid2 = int(qid1), int(qid2)

            img1 = Image.open(os.path.join(image_dir, qtype1, f'{qid1}.jpg'))
            img1 = img1.resize((224, 224))
            img2 = Image.open(os.path.join(image_dir, qtype1, f'{qid2}.jpg'))
            img2 = img2.resize((224, 224))

            prefix = 'summarize:'
            # text1 = prefix + 'a photo of ' + statement1
            # text2 = prefix + 'a photo of ' + statement2
            text1 = prefix + statement1
            text2 = prefix + statement2

            text1 = tokenizer(text1, return_tensors='pt', max_length=80,
                      truncation=True, padding='max_length').input_ids.cuda()
            text2 = tokenizer(text2, return_tensors='pt', max_length=80,
                      truncation=True, padding='max_length').input_ids.cuda()

            img1 = preprocess(images=img1, return_tensors='pt').pixel_values.to(torch.float16).cuda()
            img2 = preprocess(images=img2, return_tensors='pt').pixel_values.to(torch.float16).cuda()
            imgs = torch.cat((img1, img2), dim=0)

            with torch.no_grad():
                logits_per_image1, logits_per_text1 = model(image=imgs, text=text1, mode=model_name)
                logits_per_image2, logits_per_text2 = model(image=imgs, text=text2, mode=model_name)

                probs1 = logits_per_text1.float().softmax(dim=-1).cpu().numpy()
                probs2 = logits_per_text2.float().softmax(dim=-1).cpu().numpy()

            img1_score1 = probs1[0][0]
            img1_score2 = probs2[0][0]

            pred1 = 'img1' if img1_score1 > 0.5 else 'img2'
            pred2 = 'img1' if img1_score2 > 0.5 else 'img2'

            gt1 = 'img1' if qid1 % 2 == 1 else 'img2'
            gt2 = 'img1' if qid2 % 2 == 1 else 'img2'

            csv_writer.writerow([qid1, qid2, pred1, pred2, gt1, gt2, img1_score1, img1_score2])

            current_category = categories[num_pairs // 15]
            if pred1 == gt1 and pred2 == gt2:
                pair_accuracies[current_category] += 1
            num_pairs += 1

        csv_outfile.close()

    # Calculate percentage accuracies
    for category in pair_accuracies:
        pair_accuracies[category] = (pair_accuracies[category] / (num_pairs // len(categories))) * 100

    return pair_accuracies


parser = argparse.ArgumentParser(description='Process a directory path.')

# Adding an argument for the directory path
parser.add_argument('--directory', type=str, help='The path to the directory')

# Parsing the arguments
args = parser.parse_args()

# InternVL models
models = ['InternVL-C', 'InternVL-G']

results = {f'{model}': benchmark_model(model, args.directory) for model in models}

print(results)

# Convert results to format suitable for star plot
categories = results[list(results.keys())[0]].keys()
print(f'categories: {categories}')
data = {'Categories': list(categories)}
print(f'data: {data}')
for model in list(results.keys()):
    data[model] = [results[model][category] for category in categories]
print(f'data: {data}')
