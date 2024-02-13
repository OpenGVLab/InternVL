import argparse
import json
import os

argparse = argparse.ArgumentParser()
argparse.add_argument('--image_result_file', type=str, default='')
argparse.add_argument('--anno_path', type=str, default='data/SEED/SEED-Bench.json')

args = argparse.parse_args()
image_result_file = args.image_result_file
anno_path = args.anno_path

assert image_result_file.endswith('.jsonl')


def is_integer_string(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def filter_questions(data, task='all'):
    if task == 'image':
        return [q for q in data if 1 <= q['question_type_id'] <= 9]
    elif task == 'video':
        return [q for q in data if 10 <= q['question_type_id'] <= 12]
    elif task == 'all':
        return data
    elif is_integer_string(task):
        return [q for q in data if q['question_type_id'] == int(task)]
    else:
        raise ValueError(f'Invalid task: {task}')


if __name__ == '__main__':

    qa_anno = json.load(open(anno_path, 'rb'))
    if 'questions' in qa_anno.keys():
        question_type = qa_anno['question_type']
        question_id_type = {v: k for k, v in question_type.items()}
        qa_anno = qa_anno['questions']

    qa_anno = filter_questions(qa_anno, 'all')
    print(f'length: {len(qa_anno)}')

    with open(image_result_file, 'r') as f:

        image_result = [json.loads(line) for line in f.readlines()]

    results = []

    results.extend(image_result)

    qa_id_anno = {}
    for item in qa_anno:
        question_id = str(item['question_id'])
        qa_id_anno[question_id] = item

    type_counts = {k: [] for k, v in question_id_type.items()}

    for item in results:
        pred, gt, question_id = item['prediction'], item['answer'], item['question_id']
        question_id = str(question_id)
        question_type = qa_id_anno[question_id]['question_type_id']
        data_type = qa_id_anno[question_id]['data_type']
        gt = qa_id_anno[question_id]['answer']
        if len(pred) != 1:
            pred = pred[0]
        if pred == gt:
            type_counts[question_type].append(1)
        else:
            type_counts[question_type].append(0)

    print('Accuracy for each data type:')
    total_count, image_count, video_count = 0, 0, 0
    total_correct, image_correct, video_correct = 0, 0, 0
    for data_type_id, result in type_counts.items():
        accuracy = sum(result) / len(result) * 100
        data_type = question_id_type[data_type_id]
        print(f'Data type {data_type}: {accuracy:.2f}%')

        total_count += len(result)
        total_correct += sum(result)
        if data_type_id >= 1 and data_type_id <= 9:
            image_count += len(result)
            image_correct += sum(result)
        else:
            video_count += len(result)
            video_correct += sum(result)

    total_accuracy = total_correct / total_count * 100
    image_accuracy = image_correct / image_count * 100
    video_accuracy = video_correct / video_count * 100

    print(f'Total accuracy: {total_accuracy:.2f}%')
    print(f'Image accuracy: {image_accuracy:.2f}%')
    print(f'Video accuracy: {video_accuracy:.2f}%')
