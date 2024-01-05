import json
import logging
import math
import os
import random
import sys
import warnings
from copy import deepcopy
from typing import Dict, Optional

from internvl.conversation import SeparatorStyle, get_conv_template
from internvl.train.trainer_monkey_patch import replace_create_optimizer
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import transformers
from internvl.dist_utils import init_dist
from internvl.model.internvl_chat_with_qllama import (InternVLChatConfig,
                                                      InternVLChatModel,
                                                      InternVLConfig,
                                                      InternVLModel)
from internvl.patch import (replace_llama_attn_with_flash_attn,
                            replace_llama_rmsnorm_with_fused_rmsnorm)
from internvl.train.dataset import (TCSLoader, WeightedConcatDataset,
                                    build_transform)
from PIL import Image, ImageFile, PngImagePlugin
from torch.utils.data import Dataset
from transformers import (HfArgumentParser, LlamaConfig, LlamaForCausalLM,
                          LlamaTokenizer, Trainer, TrainingArguments,
                          default_data_collator, set_seed)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry
from transformers.utils.logging import (enable_default_handler,
                                        enable_explicit_format, set_verbosity)

replace_llama_attn_with_flash_attn()
replace_llama_rmsnorm_with_fused_rmsnorm()

try:
    from petrel_client.client import Client
    from petrel_client.common.config import Config
except ImportError as E:
    print('please install petrel_client')

IGNORE_INDEX = -100
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN = '<IMG>'
IMG_END_TOKEN = '</IMG>'
QUERY_CONTEXT_TOKEN = '<QUERY_CONTEXT>'
QUERY_START_TOKEN = '<QUERY>'
QUERY_END_TOKEN = '</QUERY>'

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    internvl_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    llm_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    freeze_llm: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the LLM decoder.'},
    )
    freeze_internvl: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the InternVL (backbone & QLLaMA) model.'},
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the vision backbone of the model.'},
    )
    freeze_qllama: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the QLLaMA model.'},
    )
    freeze_mlp: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the MLP layers of the model.'},
    )
    unfreeze_crossattn: bool = field(
        default=False,
        metadata={'help': 'Set to True to unfreeze the cross-attention layers of the model.'},
    )
    unfreeze_vit_layers: int = field(
        default=0,
        metadata={'help': 'Specify the number of ViT layers to unfreeze. Default is 0.'},
    )
    use_backbone_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the backbone model. Default is 0.'}
    )
    use_qllama_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the QLLaMA component. Default is 0.'}
    )
    use_llm_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the LLM. Default is 0.'}
    )
    unfreeze_lm_head: bool = field(
        default=False,
        metadata={'help': "Set to True to unfreeze the language model's head."},
    )
    use_custom_trainer: bool = field(
        default=False,
        metadata={'help': 'Set to True to enable the use of a custom trainer.'},
    )
    grad_checkpoint: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use gradient checkpointing.'},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    max_seq_length: Optional[int] = field(
        default=2048,
        metadata={
            'help': (
                'The maximum total input sequence length after tokenization. Sequences longer '
                'than this will be truncated, sequences shorter will be padded.'
            )
        },
    )
    max_question_length: Optional[int] = field(
        default=80,
        metadata={
            'help': (
                'The maximum input question length after tokenization. Sequences longer '
                'than this will be truncated, sequences shorter will be padded.'
            )
        },
    )
    force_image_size: Optional[int] = field(
        default=224,
        metadata={'help': 'Set the desired size for the image. Default is 224.'},
    )
    pad2square: Optional[bool] = field(
        default=False,
        metadata={'help': 'Pad the image to a square shape if set to True.'},
    )
    conv_style: Optional[str] = field(
        default='vicuna_v1.1', metadata={'help': 'Prompt style for a conversation.'}
    )
    meta_path: Optional[str] = field(
        default=None,
        metadata={'help': 'The path of the meta file of datasets.'},
    )
    with_pure_text_data: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use pure text data during supervised fine-tuning.'},
    )
    max_conv_num: Optional[int] = field(
        default=10,
        metadata={'help': 'The maximum of conversations.'},
    )
    use_data_resampling: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use data resampling.'},
    )


def preprocess(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token: int,
        num_query_token: int,
        text_only: bool = False,
) -> Dict:
    conv = get_conv_template(template_name)
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'
            if sentence['from'] == 'human':
                if not text_only:
                    if '<image>' in sentence['value']:
                        sentence['value'] = sentence['value'].replace('<image>', '<image><query>')
                    else:
                        sentence['value'] = '<query>\n' + sentence['value']
                else:
                    sentence['value'] = sentence['value'].replace('<image>', '').replace('<query>', '')
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())

    image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token}{IMG_END_TOKEN}'
    query_tokens = f'{QUERY_START_TOKEN}{QUERY_CONTEXT_TOKEN * num_query_token}{QUERY_END_TOKEN}'
    new_conversations = []
    for conversation in conversations:
        conversation = conversation.replace('<image>', image_tokens)
        conversation = conversation.replace('<query>', query_tokens)
        new_conversations.append(conversation)
    conversations = new_conversations

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors='pt',
        padding='max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1] + ': '
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == '':
                break
            turn_len = len(tokenizer(turn).input_ids)

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                instruction_len -= 1

            # Ignore the user instructions
            target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                cur_len -= 1

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            logger.info(tokenizer.decode(z))
            exit()

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                logger.info(
                    f'WARNING: tokenization mismatch: {cur_len} vs. {total_len}.'
                    f' #turn = {len(turns) - 1}. (ignored)'
                )
                sys.stdout.flush()

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, template_name, meta, llm_tokenizer, internvl_tokenizer, tcs_loader,
                 num_image_token, num_query_token, image_size=224, is_train=True, pad2square=False,
                 with_pure_text_data=False, max_conv_num=10):
        super(LazySupervisedDataset, self).__init__()
        self.llm_tokenizer = llm_tokenizer
        self.internvl_tokenizer = internvl_tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token  # 256
        self.num_query_token = num_query_token  # 96
        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square

        logger.info(f'[Dataset] num_image_token: {num_image_token}')
        logger.info(f'[Dataset] num_query_token: {num_query_token}')

        if meta['annotation'].endswith('json'):
            self.raw_data = json.load(open(meta['annotation'], 'r'))
        elif meta['annotation'].endswith('jsonl'):
            self.raw_data = [json.loads(line) for line in open(meta['annotation'], 'r')]

        logger.info(f'data length before split: {len(self.raw_data)}')
        new_raw_data = []
        for item in self.raw_data:
            if 'image' in item:  # skip pure text data
                conversations = item['conversations']
                conversations = [conversations[i:i + max_conv_num] for i in range(0, len(conversations), max_conv_num)]
                for conv in conversations:
                    new_item = item.copy()
                    if '<image>' not in conv[0]['value']:
                        conv[0]['value'] = '<image>\n' + conv[0]['value']
                    new_item['conversations'] = conv
                    new_raw_data.append(new_item)
            else:
                if with_pure_text_data:
                    new_raw_data.append(item)
        self.raw_data = new_raw_data
        logger.info(f'data length after split: {len(self.raw_data)}')

        self.root = meta['root']
        self.cached_data_dict = {}
        self.tcs_loader = tcs_loader

    def __len__(self):
        return len(self.raw_data)

    def multi_modal_get_item(self, i):
        image_path = os.path.join(self.root, self.raw_data[i]['image'])
        image = self.tcs_loader(image_path)
        transform = build_transform(is_train=self.is_train, input_size=self.image_size,
                                    pad2square=self.pad2square)
        pixel_values = transform(image)
        ret = preprocess(self.template_name, [deepcopy(self.raw_data[i]['conversations'])],
                         self.llm_tokenizer, self.num_image_token, self.num_query_token)

        questions = []
        for item in self.raw_data[i]['conversations']:
            if item['from'] == 'human':
                question = item['value'].replace('<image>\n', '')
                question = question.replace('\n<image>', '')
                question = question.replace('<image>', '')
                questions.append(question)
        tokenized_questions = self.internvl_tokenizer(
            questions,
            return_tensors='pt',
            padding='max_length',
            max_length=self.internvl_tokenizer.model_max_length,
            truncation=True,
        )
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            pixel_values=pixel_values,
            question_input_ids=tokenized_questions.input_ids,
            question_attention_mask=tokenized_questions.attention_mask,
        )
        return ret

    def pure_text_get_item(self, i):
        ret = preprocess(self.template_name, [deepcopy(self.raw_data[i]['conversations'])],
                         self.llm_tokenizer, self.num_image_token, self.num_query_token,
                         text_only=True)
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            pixel_values=None,
            question_input_ids=None,
            question_attention_mask=None,
        )
        return ret

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        while True:
            try:
                if 'image' in self.raw_data[i]:
                    ret = self.multi_modal_get_item(i)
                else:
                    ret = self.pure_text_get_item(i)
                break
            except Exception as e:
                logger.info([e, self.ds_name])
                if 'image' in self.raw_data[i]:
                    print(f"Failed to load image: {self.ds_name, self.raw_data[i]['image']}")
                    sys.stdout.flush()
                i = random.randint(0, len(self.raw_data) - 1)
        return ret


def build_datasets(data_args, llm_tokenizer, internvl_tokenizer, tcs_loader, model):
    datasets = []
    lengths = []
    ds_collections = json.loads(open(data_args.meta_path).read())
    for ds_name in ds_collections.keys():
        repeat_time = ds_collections[ds_name]['repeat_time']
        for i in range(repeat_time):
            dataset = LazySupervisedDataset(
                data_args.conv_style, ds_collections[ds_name],
                llm_tokenizer, internvl_tokenizer,
                tcs_loader,
                num_image_token=model.num_image_token,
                num_query_token=model.num_query_token,
                image_size=data_args.force_image_size,
                is_train=ds_collections[ds_name]['data_augment'],
                pad2square=data_args.pad2square,
                with_pure_text_data=data_args.with_pure_text_data,
                max_conv_num=data_args.max_conv_num
            )
            dataset.ds_name = ds_name
            datasets.append(dataset)
            if data_args.use_data_resampling:
                lengths.append(math.sqrt(len(dataset)))
            else:
                lengths.append(len(dataset))
    total_length = sum(lengths)
    weights = [l / total_length for l in lengths]
    for idx, dataset in enumerate(datasets):
        if torch.distributed.get_rank() == 0:
            logger.info(f'{dataset.ds_name}: {weights[idx]}')
    train_dataset = WeightedConcatDataset(datasets, weights)
    return train_dataset


def main():
    # Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    init_dist(launcher='slurm', backend='nccl')
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry('InternVL-Chat (w/ QLLaMA)', model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
        + f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )
    logger.info(f'Training/evaluation parameters {training_args}')

    # Detecting last checkpoint and eventually continue from last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f'Output directory ({training_args.output_dir}) already exists and is not empty. '
                'Use --overwrite_output_dir to overcome.'
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change '
                'the `--output_dir` or add `--overwrite_output_dir` to train from scratch.'
            )
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model, tokenizer, and image processor
    tokenizer_path = model_args.model_name_or_path or model_args.llm_path
    llm_tokenizer = LlamaTokenizer.from_pretrained(
        tokenizer_path, add_eos_token=False, legacy=False)
    llm_tokenizer.model_max_length = data_args.max_seq_length
    token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                  QUERY_START_TOKEN, QUERY_END_TOKEN, QUERY_CONTEXT_TOKEN]
    num_new_tokens = llm_tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = llm_tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    query_context_token_id = llm_tokenizer.convert_tokens_to_ids(QUERY_CONTEXT_TOKEN)
    tcs_loader = TCSLoader('~/petreloss.conf')

    if model_args.model_name_or_path is not None:
        logger.info('Loading InternVLChatModel...')
        model = InternVLChatModel.from_pretrained(
            model_args.model_name_or_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
    else:
        logger.info('Loading InternVL...')
        internvl = InternVLModel.from_pretrained(
            model_args.internvl_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
        logger.info('Loading LLaMA...')
        llm = LlamaForCausalLM.from_pretrained(
            model_args.llm_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
        logger.info('Building InternVLChatConfig...')
        internvl_config = InternVLConfig.from_pretrained(model_args.internvl_path)
        llm_config = LlamaConfig.from_pretrained(model_args.llm_path)
        intern_chat_config = InternVLChatConfig(internvl_config.to_dict(), llm_config.to_dict(),
                                                pad2square=data_args.pad2square)
        logger.info('Building InternVLChatModel...')
        model = InternVLChatModel(intern_chat_config, internvl, llm)

    logger.info(f'Loading InternVL Tokenizer: {model_args.internvl_path}')
    internvl_tokenizer = LlamaTokenizer.from_pretrained(
        model_args.internvl_path, add_eos_token=True)
    internvl_tokenizer.model_max_length = data_args.max_question_length
    model.img_context_token_id = img_context_token_id
    model.query_context_token_id = query_context_token_id
    logger.info('Finished')

    if data_args.force_image_size != 224:
        if model.internvl.config.force_image_size != data_args.force_image_size:
            model.internvl.config.force_image_size = data_args.force_image_size
            model.config.internvl_config.force_image_size = data_args.force_image_size
            model.internvl.vision_model.resize_pos_embeddings(
                old_size=224, new_size=data_args.force_image_size, patch_size=14)
            model.num_image_token = (data_args.force_image_size // 14) ** 2

    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(llm_tokenizer))
        # input_embeddings = model.language_model.get_input_embeddings().weight.data
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        # input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        # input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

    model.config.llm_config.vocab_size = len(llm_tokenizer)
    model.language_model.config.vocab_size = len(llm_tokenizer)

    model.internvl.config.use_cache = False
    model.internvl.config.qllama_config.use_cache = False
    model.language_model.config.use_cache = False

    if model_args.grad_checkpoint:
        model.internvl.qllama.gradient_checkpointing = True
        model.internvl.qllama.model.gradient_checkpointing = True

        model.internvl.vision_model.gradient_checkpointing = True
        model.internvl.vision_model.encoder.gradient_checkpointing = True

        model.language_model.gradient_checkpointing = True
        model.language_model.model.gradient_checkpointing = True

    train_dataset = build_datasets(data_args, llm_tokenizer, internvl_tokenizer, tcs_loader, model)

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_internvl:
        model.internvl = model.internvl.eval()
        _freeze_params(model.internvl)

    if model_args.freeze_backbone:
        model.internvl.vision_model = model.internvl.vision_model.eval()
        _freeze_params(model.internvl.vision_model)

    if model_args.freeze_qllama:
        model.internvl.qllama = model.internvl.qllama.eval()
        _freeze_params(model.internvl.qllama)
        model.internvl.query_tokens.requires_grad = False

    if model_args.freeze_llm:
        model.language_model = model.language_model.eval()
        _freeze_params(model.language_model)

    if model_args.unfreeze_lm_head:
        model.language_model.lm_head.requires_grad = True

    if model_args.use_backbone_lora:
        model.internvl.wrap_backbone_lora(r=model_args.use_backbone_lora)
        model.internvl.config.use_backbone_lora = model_args.use_backbone_lora

    if model_args.use_qllama_lora:
        model.internvl.wrap_qllama_lora(r=model_args.use_qllama_lora)
        model.internvl.config.use_qllama_lora = model_args.use_qllama_lora

    if model_args.use_llm_lora:
        model.language_model.enable_input_require_grads()
        model.wrap_llm_lora(r=model_args.use_llm_lora)
        model.config.use_llm_lora = model_args.use_llm_lora

    if model_args.unfreeze_crossattn:
        for k, v in model.internvl.named_parameters():
            if 'cross_attn' in k:
                v.requires_grad = True

    if model_args.freeze_mlp:
        _freeze_params(model.mlp1)
        _freeze_params(model.mlp2)

    if model_args.unfreeze_vit_layers != 0:
        layers = model.internvl.vision_model.encoder.layers[model_args.unfreeze_vit_layers:]
        for k, v in layers.named_parameters():
            v.requires_grad = True

    # print trainable parameters
    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name)

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    # Initialize our Trainer
    if model_args.use_custom_trainer:
        replace_create_optimizer()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=llm_tokenizer,
        data_collator=default_data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics['train_samples'] = len(train_dataset)

        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()


if __name__ == '__main__':
    main()
