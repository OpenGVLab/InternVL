# This file can be downloaded from: https://www.docvqa.org/datasets/infographicvqa and https://rrc.cvc.uab.es/?ch=17&com=introduction

import argparse
import json
import os

question_ids_to_exclude = []

# answer_types = {'image span': 'Image-Span', 'question span': 'Question-Span', 'multiple spans': 'Multi-Span', 'non span': 'None span', 'list': 'List'}
answer_types = {'image span': 'Image-Span', 'question span': 'Question-Span', 'multiple spans': 'Multi-Span',
                'non span': 'None span'}
evidence_types = {'table/list': 'Table/list', 'textual': 'Text', 'photo/pciture/visual_objects': 'Visual/Layout',
                  'figure': 'Figure', 'map': 'Map'}
reasoning_requirements = {'comparison': 'Sorting', 'arithmetic': 'Arithmetic', 'counting': 'Counting'}


def save_json(file_path, data):
    with open(file_path, 'w+') as json_file:
        json.dump(data, json_file)


def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def validate_data(gtFilePath, submFilePath):
    """
    Method validate_data: validates that all files in the results folder are correct (have the correct name contents).
                            Validates also that there are no missing files in the folder.
                            If some error detected, the method raises the error
    """

    gtJson = json.load(open(gtFilePath, 'rb'))
    submJson = json.load(open(submFilePath, 'rb'))

    if 'data' not in gtJson:
        raise Exception('The GT file is not valid (no data key)')

    if 'dataset_name' not in gtJson:
        raise Exception('The GT file is not valid (no dataset_name key)')

    if isinstance(submJson, list) is False:
        raise Exception('The Det file is not valid (root item must be an array)')

    if len(submJson) != len(gtJson['data']):
        raise Exception('The Det file is not valid (invalid number of answers. Expected:' + str(
            len(gtJson['data'])) + ' Found:' + str(len(submJson)) + ')')

    gtQuestions = sorted([r['questionId'] for r in gtJson['data']])
    res_id_to_index = {int(r['questionId']): ix for ix, r in enumerate(submJson)}
    detQuestions = sorted([r['questionId'] for r in submJson])

    if ((gtQuestions == detQuestions) is False):
        raise Exception('The Det file is not valid. Question IDs must much GT')

    for gtObject in gtJson['data']:

        try:
            q_id = int(gtObject['questionId'])
            res_ix = res_id_to_index[q_id]

        except:
            raise Exception('The Det file is not valid. Question ' + str(gtObject['questionId']) + ' not present')

        else:
            detObject = submJson[res_ix]

            #            if detObject['questionId'] != gtObject['questionId'] :
            #                raise Exception("Answer #" + str(i) + " not valid (invalid question ID. Expected:" + str(gtObject['questionId']) + "Found:" + detObject['questionId'] + ")")

            if 'answer' not in detObject:
                raise Exception('Question ' + str(gtObject['questionId']) + ' not valid (no answer key)')

            if isinstance(detObject['answer'], list) is True:
                raise Exception(
                    'Question ' + str(gtObject['questionId']) + ' not valid (answer key has to be a single string)')


def evaluate_method(gtFilePath, submFilePath, evaluationParams):
    """
    Method evaluate_method: evaluate method and returns the results
        Results. Dictionary with the following values:
        - method (required)  Global method metrics. Ex: { 'Precision':0.8,'Recall':0.9 }
        - samples (optional) Per sample metrics. Ex: {'sample1' : { 'Precision':0.8,'Recall':0.9 } , 'sample2' : { 'Precision':0.8,'Recall':0.9 }
    """

    show_scores_per_answer_type = evaluationParams.answer_types

    gtJson = json.load(open(gtFilePath, 'rb'))
    submJson = json.load(open(submFilePath, 'rb'))

    res_id_to_index = {int(r['questionId']): ix for ix, r in enumerate(submJson)}

    perSampleMetrics = {}

    totalScore = 0
    row = 0

    if show_scores_per_answer_type:
        answerTypeTotalScore = {x: 0 for x in answer_types.keys()}
        answerTypeNumQuestions = {x: 0 for x in answer_types.keys()}

        evidenceTypeTotalScore = {x: 0 for x in evidence_types.keys()}
        evidenceTypeNumQuestions = {x: 0 for x in evidence_types.keys()}

        reasoningTypeTotalScore = {x: 0 for x in reasoning_requirements.keys()}
        reasoningTypeNumQuestions = {x: 0 for x in reasoning_requirements.keys()}

    for gtObject in gtJson['data']:

        q_id = int(gtObject['questionId'])
        res_ix = res_id_to_index[q_id]
        detObject = submJson[res_ix]

        if q_id in question_ids_to_exclude:
            question_result = 0
            info = 'Question EXCLUDED from the result'

        else:
            info = ''
            values = []
            for answer in gtObject['answers']:
                # preprocess both the answers - gt and prediction
                gt_answer = ' '.join(answer.strip().lower().split())
                det_answer = ' '.join(detObject['answer'].strip().lower().split())

                # dist = levenshtein_distance(answer.lower(), detObject['answer'].lower())
                dist = levenshtein_distance(gt_answer, det_answer)
                length = max(len(answer.upper()), len(detObject['answer'].upper()))
                values.append(0.0 if length == 0 else float(dist) / float(length))

            question_result = 1 - min(values)

            if (question_result < evaluationParams.anls_threshold):
                question_result = 0

            totalScore += question_result

            if show_scores_per_answer_type:
                for q_type in gtObject['answer_type']:
                    answerTypeTotalScore[q_type] += question_result
                    answerTypeNumQuestions[q_type] += 1

                for q_type in gtObject['evidence']:
                    evidenceTypeTotalScore[q_type] += question_result
                    evidenceTypeNumQuestions[q_type] += 1

                for q_type in gtObject['operation/reasoning']:
                    reasoningTypeTotalScore[q_type] += question_result
                    reasoningTypeNumQuestions[q_type] += 1

        perSampleMetrics[str(gtObject['questionId'])] = {
            'score': question_result,
            'question': gtObject['question'],
            'gt': gtObject['answers'],
            'det': detObject['answer'],
            'info': info
        }
        row = row + 1

    methodMetrics = {
        'score': 0 if len(gtJson['data']) == 0 else totalScore / (len(gtJson['data']) - len(question_ids_to_exclude))
    }

    answer_types_scores = {}
    evidence_types_scores = {}
    operation_types_scores = {}

    if show_scores_per_answer_type:
        for a_type, ref in answer_types.items():
            answer_types_scores[ref] = 0 if len(gtJson['data']) == 0 else answerTypeTotalScore[a_type] / (
            answerTypeNumQuestions[a_type])

        for e_type, ref in evidence_types.items():
            evidence_types_scores[ref] = 0 if len(gtJson['data']) == 0 else evidenceTypeTotalScore[e_type] / (
            evidenceTypeNumQuestions[e_type])

        for r_type, ref in reasoning_requirements.items():
            operation_types_scores[ref] = 0 if len(gtJson['data']) == 0 else reasoningTypeTotalScore[r_type] / (
            reasoningTypeNumQuestions[r_type])

    resDict = {
        'result': methodMetrics,
        'scores_by_types': {'answer_types': answer_types_scores, 'evidence_types': evidence_types_scores,
                            'operation_types': operation_types_scores},
        'per_sample_result': perSampleMetrics
    }

    return resDict


def display_results(results, show_answer_types):
    print('\nOverall ANLS: {:2.4f}'.format(results['result']['score']))

    if show_answer_types:
        print('\nAnswer types:')
        for a_type in answer_types.values():
            print('\t{:12s} {:2.4f}'.format(a_type, results['scores_by_types']['answer_types'][a_type]))

        print('\nEvidence types:')
        for e_type in evidence_types.values():
            print('\t{:12s} {:2.4f}'.format(e_type, results['scores_by_types']['evidence_types'][e_type]))

        print('\nOperation required:')
        for r_type in reasoning_requirements.values():
            print('\t{:12s} {:2.4f}'.format(r_type, results['scores_by_types']['operation_types'][r_type]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='InfographVQA evaluation script.')

    parser.add_argument('-g', '--ground_truth', type=str, help='Path of the Ground Truth file.', required=True)
    parser.add_argument('-s', '--submission_file', type=str, help="Path of your method's results file.", required=True)

    parser.add_argument('-t', '--anls_threshold', type=float, default=0.5,
                        help='ANLS threshold to use (See Scene-Text VQA paper for more info.).', required=False)
    parser.add_argument('-a', '--answer_types', type=bool, default=False,
                        help='Score break down by answer types (special gt file required).', required=False)
    parser.add_argument('-o', '--output', type=str,
                        help="Path to a directory where to copy the file 'results.json' that contains per-sample results.",
                        required=False)

    args = parser.parse_args()

    # Validate the format of ground truth and submission files.
    validate_data(args.ground_truth, args.submission_file)

    # Evaluate method
    results = evaluate_method(args.ground_truth, args.submission_file, args)

    display_results(results, args.answer_types)

    if args.output:
        output_dir = args.output

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        resultsOutputname = os.path.join(output_dir, 'results.json')
        save_json(resultsOutputname, results)

        print('All results including per-sample result has been correctly saved!')
