"""Parse and Evalate"""
import json
from argparse import ArgumentParser

from data_utils import CAT_SHORT2LONG, DOMAIN_CAT2SUB_CAT, save_json
from eval_utils import calculate_ins_level_acc, evaluate, parse_open_response

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--output_path', type=str, default='./example_outputs/qwen_vl/total_val_output.json',
                        help='The path to model output file.')
    parser.add_argument('--answer_path', type=str, default='./answer_dict_val.json', help='Answer file path.')
    args = parser.parse_args()

    output_dict = json.load(open(args.output_path))
    answer_dict = json.load(open(args.answer_path))

    # group by category
    output_dict_w_cat = {}
    for data_id, parsed_pred in output_dict.items():
        category = '_'.join(data_id.split('_')[1:-1])
        if category not in output_dict_w_cat:
            output_dict_w_cat.update({category: {}})
        output_dict_w_cat[category].update({data_id: parsed_pred})

    # group by category
    answer_dict_w_cat = {}
    for data_id, parsed_pred in answer_dict.items():
        category = '_'.join(data_id.split('_')[1:-1])
        if category not in answer_dict_w_cat:
            answer_dict_w_cat.update({category: {}})
        answer_dict_w_cat[category].update({data_id: parsed_pred})

    evaluation_result = {}

    for category in CAT_SHORT2LONG.values():
        print('Evaluating: {}'.format(category))
        # get cat_outputs and cat_answers
        try:
            cat_outputs = output_dict_w_cat[category]
            cat_answers = answer_dict_w_cat[category]
        except KeyError:
            print('Skipping {} for not found'.format(category))
            continue

        exampels_to_eval = []
        for data_id, parsed_pred in cat_outputs.items():
            question_type = cat_answers[data_id]['question_type']
            if question_type != 'multiple-choice':
                parsed_pred = parse_open_response(parsed_pred)  # mainly for type consistency (make it number, etc.)
            else:
                parsed_pred = parsed_pred

            exampels_to_eval.append({
                'id': data_id,
                'question_type': question_type,
                'answer': cat_answers[data_id]['ground_truth'],
                'parsed_pred': parsed_pred
            })

        judge_dict, metric_dict = evaluate(exampels_to_eval)
        metric_dict.update({'num_example': len(exampels_to_eval)})

        evaluation_result[category] = metric_dict

    printable_results = {}
    # pdb.set_trace()
    # add domain Subject
    for domain, in_domain_cats in DOMAIN_CAT2SUB_CAT.items():
        in_domain_cat_results = {}
        for cat_name in in_domain_cats:  # use the order in DOMAIN_CAT2SUB_CAT
            if cat_name in evaluation_result.keys():
                in_domain_cat_results[cat_name] = evaluation_result[cat_name]
            else:
                pass
        in_domain_ins_acc = calculate_ins_level_acc(in_domain_cat_results)
        in_domain_data_num = sum([cat_results['num_example'] for cat_results in in_domain_cat_results.values()])
        printable_results['Overall-' + domain] = {'num': int(in_domain_data_num),
                                                  'acc': round(in_domain_ins_acc, 3)
                                                  }
        # add sub category
        for cat_name, cat_results in in_domain_cat_results.items():
            printable_results[cat_name] = {'num': int(cat_results['num_example']),
                                           'acc': round(cat_results['acc'], 3)
                                           }

    # table.append(["-----------------------------", "-----", "----"])
    all_ins_acc = calculate_ins_level_acc(evaluation_result)
    printable_results['Overall'] = {
        'num': sum([cat_results['num_example'] for cat_results in evaluation_result.values()]),
        'acc': round(all_ins_acc, 3)}

    print(printable_results)
