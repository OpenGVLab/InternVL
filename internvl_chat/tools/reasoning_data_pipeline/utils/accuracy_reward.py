import re

from sympy import latex
from sympy.parsing.latex import parse_latex
from tqdm import tqdm


class EvalAIAnswerProcessor:
    """
    Processes an answer similar to Eval AI
        copied from
        https://github.com/facebookresearch/mmf/blob/c46b3b3391275b4181567db80943473a89ab98ab/pythia/tasks/processors.py#L897
    """

    CONTRACTIONS = {
        'aint': "ain't",
        'arent': "aren't",
        'cant': "can't",
        'couldve': "could've",
        'couldnt': "couldn't",
        "couldn'tve": "couldn't've",
        "couldnt've": "couldn't've",
        'didnt': "didn't",
        'doesnt': "doesn't",
        'dont': "don't",
        'hadnt': "hadn't",
        "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've",
        'hasnt': "hasn't",
        'havent': "haven't",
        'hed': "he'd",
        "hed've": "he'd've",
        "he'dve": "he'd've",
        'hes': "he's",
        'howd': "how'd",
        'howll': "how'll",
        'hows': "how's",
        "Id've": "I'd've",
        "I'dve": "I'd've",
        'Im': "I'm",
        'Ive': "I've",
        'isnt': "isn't",
        'itd': "it'd",
        "itd've": "it'd've",
        "it'dve": "it'd've",
        'itll': "it'll",
        "let's": "let's",
        'maam': "ma'am",
        'mightnt': "mightn't",
        "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've",
        'mightve': "might've",
        'mustnt': "mustn't",
        'mustve': "must've",
        'neednt': "needn't",
        'notve': "not've",
        'oclock': "o'clock",
        'oughtnt': "oughtn't",
        "ow's'at": "'ow's'at",
        "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at",
        'shant': "shan't",
        "shed've": "she'd've",
        "she'dve": "she'd've",
        "she's": "she's",
        'shouldve': "should've",
        'shouldnt': "shouldn't",
        "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've",
        "somebody'd": 'somebodyd',
        "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've",
        'somebodyll': "somebody'll",
        'somebodys': "somebody's",
        'someoned': "someone'd",
        "someoned've": "someone'd've",
        "someone'dve": "someone'd've",
        'someonell': "someone'll",
        'someones': "someone's",
        'somethingd': "something'd",
        "somethingd've": "something'd've",
        "something'dve": "something'd've",
        'somethingll': "something'll",
        'thats': "that's",
        'thered': "there'd",
        "thered've": "there'd've",
        "there'dve": "there'd've",
        'therere': "there're",
        'theres': "there's",
        'theyd': "they'd",
        "theyd've": "they'd've",
        "they'dve": "they'd've",
        'theyll': "they'll",
        'theyre': "they're",
        'theyve': "they've",
        'twas': "'twas",
        'wasnt': "wasn't",
        "wed've": "we'd've",
        "we'dve": "we'd've",
        'weve': "we've",
        'werent': "weren't",
        'whatll': "what'll",
        'whatre': "what're",
        'whats': "what's",
        'whatve': "what've",
        'whens': "when's",
        'whered': "where'd",
        'wheres': "where's",
        'whereve': "where've",
        'whod': "who'd",
        "whod've": "who'd've",
        "who'dve": "who'd've",
        'wholl': "who'll",
        'whos': "who's",
        'whove': "who've",
        'whyll': "why'll",
        'whyre': "why're",
        'whys': "why's",
        'wont': "won't",
        'wouldve': "would've",
        'wouldnt': "wouldn't",
        "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've",
        'yall': "y'all",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
        'youd': "you'd",
        "youd've": "you'd've",
        "you'dve": "you'd've",
        'youll': "you'll",
        'youre': "you're",
        'youve': "you've",
    }

    NUMBER_MAP = {
        'none': '0',
        'zero': '0',
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'ten': '10',
    }
    ARTICLES = ['a', 'an', 'the']
    PERIOD_STRIP = re.compile(r'(?!<=\d)(\.)(?!\d)')
    COMMA_STRIP = re.compile(r'(?<=\d)(\,)+(?=\d)')
    PUNCTUATIONS = [
        ';',
        r'/',
        '[',
        ']',
        '"',
        '{',
        '}',
        '(',
        ')',
        '=',
        '+',
        '\\',
        '_',
        '-',
        '>',
        '<',
        '@',
        '`',
        ',',
        '?',
        '!',
    ]

    def __init__(self, *args, **kwargs):
        pass

    def word_tokenize(self, word):
        word = word.lower()
        word = word.replace(',', '').replace('?', '').replace("'s", " 's")
        return word.strip()

    def process_punctuation(self, in_text):
        out_text = in_text
        for p in self.PUNCTUATIONS:
            if (p + ' ' in in_text or ' ' + p in in_text) or (
                re.search(self.COMMA_STRIP, in_text) is not None
            ):
                out_text = out_text.replace(p, '')
            else:
                out_text = out_text.replace(p, ' ')
        out_text = self.PERIOD_STRIP.sub('', out_text, re.UNICODE)
        return out_text

    def process_digit_article(self, in_text):
        out_text = []
        temp_text = in_text.lower().split()
        for word in temp_text:
            word = self.NUMBER_MAP.setdefault(word, word)
            if word not in self.ARTICLES:
                out_text.append(word)
            else:
                pass
        for word_id, word in enumerate(out_text):
            if word in self.CONTRACTIONS:
                out_text[word_id] = self.CONTRACTIONS[word]
        out_text = ' '.join(out_text)
        return out_text

    def __call__(self, item):
        item = self.word_tokenize(item)
        item = item.replace('\n', ' ').replace('\t', ' ').strip()
        item = self.process_punctuation(item)
        item = self.process_digit_article(item)
        return item


class TextVQAAccuracyEvaluator:
    def __init__(self):
        self.answer_processor = EvalAIAnswerProcessor()

    def _compute_answer_scores(self, raw_answers):
        """
        compute the accuracy (soft score) of human answers
        """
        answers = [self.answer_processor(a) for a in raw_answers]
        assert len(answers) == 10
        gt_answers = list(enumerate(answers))
        unique_answers = set(answers)
        unique_answer_scores = {}

        for unique_answer in unique_answers:
            accs = []
            for gt_answer in gt_answers:
                other_answers = [item for item in gt_answers if item != gt_answer]
                matching_answers = [
                    item for item in other_answers if item[1] == unique_answer
                ]
                acc = min(1, float(len(matching_answers)) / 3)
                accs.append(acc)
            unique_answer_scores[unique_answer] = sum(accs) / len(accs)

        return unique_answer_scores

    def eval_pred_list(self, pred_list, disable_tqdm=False):
        pred_scores = []
        for entry in tqdm(pred_list, disable=disable_tqdm):
            pred_answer = self.answer_processor(entry['pred_answer'])
            unique_answer_scores = self._compute_answer_scores(entry['gt_answers'])
            score = unique_answer_scores.get(pred_answer, 0.0)
            pred_scores.append(score)

        accuracy = sum(pred_scores) / len(pred_scores)
        return accuracy


evaluator_cache = {}
evaluator = TextVQAAccuracyEvaluator()
option_candidate = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']


def isfloat(x):
    try:
        float(x)
        return True
    except:
        return False


def math_score(prediction: str, target: str, max_relative_change: float = 1e-3) -> bool:
    def _to_float(text: str) -> float:
        text = text.replace('degrees', '')
        text = text.replace('degree', '')
        text = text.replace('\\angle', '')
        text = text.replace('degrees', '')
        text = text.replace('°', '')
        text = text.replace('%', '')
        text = text.replace('cm', '')

        try:
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

    def _to_float(text: str) -> float:
        try:
            if text.endswith('%'):
                # Convert percentages to floats.
                # return float(text.rstrip('%')) / 100.0
                return float(text.rstrip('%'))
            else:
                return float(text)
        except ValueError:
            return None

    if len(target) == 4 and target.startswith('20'):
        return prediction.lower() == target.lower()

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float -
                              target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


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


def multi_choice_score(answer_pred, answer_gt):
    answer_pred = answer_pred.strip()
    answer_gt = answer_gt.strip()
    if answer_pred.lower() == answer_gt.lower():
        return 1

    if len(answer_pred) >= 2 and answer_pred[1] == '.':
        answer_pred = answer_pred[0]

    if len(answer_pred) >= 3 and answer_pred[0] == '(' and answer_pred[2] == ')':
        answer_pred = answer_pred[1]

    return answer_pred.lower() == answer_gt.lower()


def parse_answer(response, prompt_version):
    if prompt_version in ['zh', 'en']:
        return extract_answer_from_mpo(response, version=prompt_version)
    if prompt_version in ['zh_v2', 'en_v2', 'en_r1', 'zh_r1']:
        return None, extract_answer_from_box(response)
    raise NotImplementedError(f'Unsupported prompt_version: {prompt_version}')


def extract_answer_from_mpo(response, version):
    if version == 'en':
        answer_trigger = 'Final answer:'
    elif version == 'zh':
        answer_trigger = '答案:'
    else:
        raise NotImplementedError(f'Unsupported prompt version {version}')

    answer_trigger = 'Final answer:'
    if response.count(answer_trigger) == 0:
        answer_trigger = 'Final Answer:'
    if response.count(answer_trigger) == 0:
        answer_trigger = '答案:'

    assert response.count(answer_trigger) <= 2, f'Fail to find Answer, {response.count(answer_trigger)=}'
    assert response.count('\n') >= 2, f'Fail to find rationale, {response=}'

    rationale, answer = response.rsplit(answer_trigger, 1)
    assert len(rationale.strip()) > 0, f'Empty rationale:\n{response}'
    assert '\n' not in answer.strip(), f'Answer with multiple paragraphs:\n{answer}'

    return rationale.strip(), answer.strip()


def extract_answer_from_box(ans):
    idx = ans.rfind(r'\boxed{')
    if idx == -1:
        return ans

    idx += len(r'\boxed{')
    brace_level = 1
    content_start = idx
    i = idx

    while i < len(ans):
        if ans[i] == '{':
            brace_level += 1
        elif ans[i] == '}':
            brace_level -= 1
            if brace_level == 0:
                break
        i += 1

    if brace_level != 0:
        # Unbalanced braces
        return ans

    content = ans[content_start:i]
    return content


def check_answer(answer_pred, answer_gt, mode):
    if (answer_pred, answer_gt) in evaluator_cache:
        accuracy = evaluator_cache[(answer_pred, answer_gt)]

    if answer_pred.lower() == answer_gt.lower():
        return 1

    accuracy = 0

    # vqa_score
    if 'vqa_score' in mode:
        merged_outputs = [
            {
                'pred_answer': answer_pred,
                'gt_answers': [answer_gt] * 10,
            },
        ]
        accuracy = max(accuracy, evaluator.eval_pred_list(merged_outputs, disable_tqdm=True))

        if len(evaluator.answer_processor(answer_pred)) == 0:
            accuracy = 0

        if len(evaluator.answer_processor(answer_gt)) == 0:
            accuracy = 0

    # relaxed_accuracy (e.g. charqa)
    if 'relaxed_accuracy' in mode:
        accuracy = max(accuracy, relaxed_correctness(answer_gt, answer_pred))

    # anls (e.g. docvqa, infographicsvqa)
    if 'anls' in mode:
        gt_answer = ' '.join(answer_gt.strip().lower().split())
        det_answer = ' '.join(answer_pred.strip().lower().split())
        dist = levenshtein_distance(gt_answer, det_answer)
        length = max(len(answer_gt.upper()), len(answer_pred.upper()))
        accuracy = max(accuracy, float(dist) / float(length))

    if 'mc_score' in mode:
        accuracy = max(accuracy, multi_choice_score(answer_pred, answer_gt))

    if 'math_score' in mode:
        accuracy = max(accuracy, math_score(answer_pred, answer_gt))

    if 'latex_score' in mode and (use_latex_score(answer_pred) or use_latex_score(answer_gt)):
        accuracy = max(accuracy, latex_score(answer_pred, answer_gt))

    accuracy = int(accuracy > 0.9)
    evaluator_cache[(answer_pred, answer_gt)] = accuracy
    return accuracy


def fix_answer(response, answer_pred, answer_gt):
    answer_pred_orig = answer_pred
    answer_gt_orig = answer_gt
    answer_pred = answer_pred.lower()
    answer_gt = answer_gt.lower()

    if answer_gt.upper() in option_candidate:
        try:
            answer_pred = post_process(answer_pred_orig)
        except:
            return response

        answer_gt = answer_gt.upper()

    if (
        answer_gt in answer_pred
        # 30,594 -> 30594
        or answer_gt.strip('.').replace(',', '') in answer_pred.strip('.').replace(',', '')
    ):
        response = answer_gt_orig.join(response.rsplit(answer_pred_orig, 1))
        response = response.strip().strip('**').strip()

    other_lines, last_line = response.rsplit('\n', 1)
    if '**Final' in last_line:
        last_line = last_line.replace('**Final', 'Final')
        response = f'{other_lines}\n{last_line}'.strip()

    return response


def contain_keywords(ds_name, keywords):
    for keyword in keywords:
        if keyword in ds_name:
            return True
    return False


def post_process(pred):
    pred = pred.strip().strip('*').strip().upper()

    if len(pred) == 1:
        return pred

    if len(pred) > 1 and not pred[1].isalpha() and pred[0] in option_candidate:
        return pred[0]

    if len(pred) > 2 and pred[0] == '(' and pred[2] == ')' and pred[1] in option_candidate:
        return pred[1]

    raise RuntimeError(f'Fail to parse pred: {pred}')


def get_mode(ds_name):
    if contain_keywords(ds_name, ['chartqa']):
        return ['relaxed_accuracy']

    if contain_keywords(ds_name, ['docvqa', 'infographics']):
        return ['anls']

    if contain_keywords(ds_name, ['SROIE', 'CLEVR_math', 'geos', 'geometry']):
        return ['relaxed_accuracy', 'vqa_score', 'mc_score']

    if contain_keywords(ds_name, ['mavis']):
        return ['vqa_score', 'mc_score', 'math_score', 'latex_score']

    return ['vqa_score', 'mc_score', 'math_score']


def use_latex_score(x):
    # \\frac{1}{2}
    if '\\' in x:
        return True

    # 1/2, 2/4
    PATTERN = r'^\s*\d+\s*/\s*\d+\s*$'
    if bool(re.match(PATTERN, x)):
        return True

    return False


def validate_latex(pred, gt, easy_mode=False):
    if pred == gt:
        return True
    try:
        pred = parse_latex(pred)
        gt = parse_latex(gt)
    except:
        return False

    if easy_mode:
        funcs = [
            lambda x: x,
        ]
    else:
        funcs = [
            lambda x: x,
            lambda x: x.simplify(),
            lambda x: parse_latex(latex(x)),
            lambda x: x.expand(),
            lambda x: x.factor(),
            lambda x: x.nsimplify(),
            lambda x: x.nsimplify().simplify(),
            lambda x: x.trigsimp(),
        ]

    for i in range(len(funcs)):
        for j in range(len(funcs)):
            try:
                expr1 = funcs[i](pred)
                expr2 = funcs[j](gt)

                if expr1.equals(expr2):
                    return True
            except:
                pass
    return False


def latex_score(prediction, target):
    return float(validate_latex(prediction, target, easy_mode=True))
