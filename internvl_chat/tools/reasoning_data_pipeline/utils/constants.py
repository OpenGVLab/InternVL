IMG_PLACEHOLDER = '<image>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'


R1_SYSTEM_PROMPT = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.
""".strip()


PRM_SYSTEM_PROMPT = """
You are an advanced AI assistant, designed to serve as a process supervision model. In this task, I will provide a problem statement followed by the first step of the solution process. For each subsequent turn, I will give you a new step in the solution. Your role is to assess whether the solution process is correct up to the current step.

- In the **first round**, I will input the problem and the first step of the solution process.
- In **each subsequent round**, I will provide the next step in the solution.

For each step, you should:

- Respond with **"+"** if you believe the solution process is correct up to this step.
- Respond with **"-"** if you detect any issues or errors in the process up to this step.

Please note:
- Only respond with **"+"** or **"-"**. Do not provide any additional explanations, comments, or justifications.

Your task is to verify the accuracy and correctness of each step in the given solution process.
""".strip()


INSTRUCTION_EN = (
    'Your task is to answer the question below. '
    "Give step by step reasoning before you answer, and when you're ready to answer, "
    "please use the format \"Final answer: ..\""
    '\n\n'
    'Question:'
    '\n\n'
    '{question}'
)


INSTRUCTION_BOXED_EN = (
    '{question}\n'
    'Answer the preceding question. The last line of your response should follow this format: '
    "'Answer: \\boxed{$FINAL_ANSWER}' (without quotes), where 'FINAL_ANSWER' is your conclusion "
    'based on the reasoning provided. If you are uncertain or the problem is too complex, make '
    'a reasoned guess based on the information provided. Avoid repeating steps indefinitely—'
    'provide your best guess even if unsure. Think step by step logically, considering all '
    'relevant information before answering.'
)


INSTRUCTION_R1_EN = R1_SYSTEM_PROMPT + '\n' + '{question}\nPlease answer the question and put the final answer within \\boxed{}.'


INSTRUCTION_ZH = (
    "你的任务是回答以下问题。在回答之前，请逐步推理说明您的思路。当你准备好给出答案时，请使用以下格式：\"答案: ...\""
    '\n\n'
    '问题:'
    '\n\n'
    '{question}'
)


INSTRUCTION_BOXED_ZH = (
    '{question}\n'
    '请回答上述问题。你的回答最后一行应遵循以下格式：'
    '“答案：\\boxed{$FINAL_ANSWER}”（不包含引号），其中“FINAL_ANSWER”是你根据推理得出的最终结论。'
    '如果你不确定答案或问题过于复杂，请基于已有信息做出有理有据的猜测。'
    '避免无限重复推理步骤——即使不完全确定，也请给出你认为最合理的答案。'
    '请一步一步进行逻辑思考，充分考虑所有相关信息后再作答。'
)


INSTRUCTION_R1_ZH = R1_SYSTEM_PROMPT + '\n' + '{question}\n请详细回答这个问题并将最终答案用\\boxed{}包住。'


VALID_INSTRUCTIONS = [
    'Answer the question using a single word or phrase.',
    "Answer with the option's letter from the given choices directly.",
    'Please answer Yes or No.',
]
VALID_INSTRUCTIONS = set(VALID_INSTRUCTIONS)
