import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch.distributed as dist
import transformers
from internvl.dist_utils import init_dist
from internvl.model.internvl_stage2_retrieval import (InternVLConfig,
                                                      InternVLModel)
from internvl.train.dataset import COCODataset, FlickrDataset
from internvl.train.trainer_monkey_patch import replace_create_optimizer
from PIL import Image, ImageFile, PngImagePlugin
from transformers import (HfArgumentParser, LlamaTokenizer, Trainer,
                          TrainingArguments, default_data_collator, set_seed)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import (enable_default_handler,
                                        enable_explicit_format, set_verbosity)

IGNORE_INDEX = -100
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

ds_collections = {
    'flickr30k_en_train': {
        'root': './data/flickr30k/Images/',
        'annotation': './data/flickr30k/flickr30k_train_karpathy.txt',
    },
    'flickr30k_cn_train': {
        'root': './data/flickr30k/Images/',
        'annotation': './data/flickr30k/flickr30k_cn_train.txt',
    },
    'coco_karpathy_train': {
        'root': './data/coco/',
        'annotation': './data/coco/annotations/coco_karpathy_train.json',
    },
}


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    freeze_model: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the entire model.'},
    )
    freeze_vision_model: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the vision backbone of the model.'},
    )
    freeze_qllama: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the QLLaMA of the model.'},
    )
    unfreeze_qllama_head: bool = field(
        default=False,
        metadata={'help': 'Set to True to unfreeze the head of the QLLaMA.'},
    )
    unfreeze_crossattn: bool = field(
        default=False,
        metadata={'help': 'Set to True to unfreeze the cross attention layers in the QLLaMA.'},
    )
    use_backbone_lora: int = field(
        default=0, metadata={'help': 'If non-zero, indicates the use of LoRA in the vision backbone of the model'}
    )
    use_qllama_lora: int = field(
        default=0, metadata={'help': 'If non-zero, indicates the use of LoRA in the QLLaMA of the model'}
    )
    use_custom_trainer: bool = field(
        default=False, metadata={'help': 'Set to True to enable the use of a custom trainer.'},
    )
    drop_path_rate: float = field(
        default=0.0, metadata={'help': 'Specify the value of drop path rate in the vision backbone. Default is 0.'}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default='flickr30k_en_train',
        metadata={'help': 'Specify the name of dataset to be used.'},
    )
    max_seq_length: Optional[int] = field(
        default=80,
        metadata={
            'help': (
                'The maximum total input sequence length after tokenization. Sequences longer '
                'than this will be truncated, sequences shorter will be padded.'
            )
        },
    )
    force_image_size: Optional[int] = field(
        default=224,
        metadata={'help': 'Specify the image size for training models.'},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            'help': (
                'Whether to pad all samples to model maximum sentence length. '
                'If False, will pad the samples dynamically when batching to the maximum length in the batch. More '
                'efficient on GPU but very bad for TPU.'
            )
        },
    )


def main():
    # Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # If use DeepSpeed zero3, init_dist must before HfArgumentParser
    launcher = os.environ.get('LAUNCHER', 'slurm')
    init_dist(launcher=launcher, backend='nccl')
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry('finetune Flickr30K', model_args, data_args)

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
    tokenizer = LlamaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        add_eos_token=True
    )

    if 'flickr' in data_args.dataset_name:
        train_dataset = FlickrDataset(metas=ds_collections[data_args.dataset_name],
                                      tokenizer=tokenizer, data_args=data_args)
    elif 'coco' in data_args.dataset_name:
        train_dataset = COCODataset(metas=ds_collections[data_args.dataset_name],
                                    tokenizer=tokenizer, data_args=data_args)
    config = InternVLConfig.from_pretrained(model_args.model_name_or_path)
    config.vision_config.drop_path_rate = model_args.drop_path_rate
    model = InternVLModel.from_pretrained(
        model_args.model_name_or_path,
        # ignore_mismatched_sizes=True,
        config=config
    )
    if data_args.force_image_size != 224:
        model.config.force_image_size = data_args.force_image_size
        model.vision_model.resize_pos_embeddings(old_size=224, new_size=data_args.force_image_size, patch_size=14)

    model.config.use_cache = False
    model.config.qllama_config.use_cache = False
    model.qllama.gradient_checkpointing = True
    model.qllama.model.gradient_checkpointing = True
    model.vision_model.gradient_checkpointing = True
    model.vision_model.encoder.gradient_checkpointing = True

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_model:
        _freeze_params(model)

    if model_args.freeze_vision_model:
        model.vision_model = model.vision_model.eval()
        _freeze_params(model.vision_model)

    if model_args.freeze_qllama:
        model.qllama = model.qllama.eval()
        _freeze_params(model.qllama)

    if model_args.use_backbone_lora:
        model.wrap_backbone_lora(r=model_args.use_backbone_lora, lora_alpha=model_args.use_backbone_lora * 2)
        model.config.use_backbone_lora = model_args.use_backbone_lora

    if model_args.use_qllama_lora:
        model.wrap_qllama_lora(r=model_args.use_qllama_lora, lora_alpha=model_args.use_backbone_lora * 2)
        model.config.use_qllama_lora = model_args.use_qllama_lora

    if model_args.unfreeze_crossattn:
        for name, param in model.qllama.named_parameters():
            if 'cross_attn' in name:
                param.requires_grad = True

    if model_args.unfreeze_qllama_head:
        model.qllama.lm_head.weight.requires_grad = True
        model.text_projection.requires_grad = True

    # print trainable parameters
    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            print(name, param.requires_grad)

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    # Initialize our Trainer
    if model_args.use_custom_trainer:
        replace_create_optimizer()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
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
