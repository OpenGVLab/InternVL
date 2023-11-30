import json
import os

import torch
import torch.nn as nn
import transformers
from transformers import Trainer, logging
from transformers.trainer import is_sagemaker_mp_enabled

logger = logging.get_logger(__name__)


def get_num_layer_for_vit_and_qllama(var_name, vit_num_max_layer, llama_num_max_layer):
    if var_name in ('query_tokens', 'logit_scale',):
        return 0
    if var_name.startswith('clip_projector.'):
        return vit_num_max_layer
    if var_name.startswith('clip_projector2.') or var_name.startswith('itm_head.') or \
            var_name == 'text_projection':
        return llama_num_max_layer
    if var_name.startswith('vision_model.'):
        if 'embeddings.' in var_name:
            return 0
        if 'layers.' in var_name:
            var_name = var_name.split('layers.')[-1]
            layer_id = int(var_name.split('.')[0])
            return layer_id + 1
    if var_name.startswith('qllama.'):
        if 'embed_tokens' in var_name:
            return 0
        if 'layers.' in var_name:
            var_name = var_name.split('layers.')[-1]
            layer_id = int(var_name.split('.')[0])
            return layer_id + 1
        else:
            return llama_num_max_layer
    return 0


def param_classification(name):
    if name in ['query_tokens', 'text_projection', 'logit_scale']:
        return 'qllama'
    elif name.startswith('vision_model.'):
        return 'vit'
    elif name.startswith('qllama.'):
        return 'qllama'
    elif name.startswith('clip_projector.'):
        return 'vit'
    elif name.startswith('clip_projector2.'):
        return 'qllama'
    elif name.startswith('itm_head.'):
        return 'qllama'
    else:
        return 'other'


def create_optimizer(self):
    """
    Setup the optimizer.

    We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    Trainer's init through `optimizers`, or subclass and override this method in a subclass.
    """
    opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

    parameter_groups = {}
    try:  # for stage2 model
        vit_num_layers = opt_model.config.vision_config.num_hidden_layers + 2
        qllama_num_layers = opt_model.config.qllama_config.num_hidden_layers + 2
    except:  # for stage3 model
        vit_num_layers = opt_model.qllama.config.vision_config.num_hidden_layers + 2
        qllama_num_layers = opt_model.qllama.config.qllama_config.num_hidden_layers + 2
    print('vit_num_layers:', vit_num_layers)
    print('qllama_num_layers:', qllama_num_layers)

    vit_layer_decay_rate = float(os.getenv('VIT_LAYER_DECAY_RATE', 1.0))
    qllama_layer_decay_rate = float(os.getenv('QLLAMA_LAYER_DECAY_RATE', 1.0))
    print('vit_layer_decay_rate:', vit_layer_decay_rate)
    print('qllama_layer_decay_rate:', qllama_layer_decay_rate)

    for name, param in opt_model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith('.bias'):
            group_name = 'no_decay'
            this_weight_decay = 0.
        else:
            group_name = 'decay'
            this_weight_decay = self.args.weight_decay

        cls = param_classification(name)
        layer_id = get_num_layer_for_vit_and_qllama(name, vit_num_layers, qllama_num_layers)
        group_name = '%s_layer_%d_%s' % (cls, layer_id, group_name)
        if group_name not in parameter_groups:
            if cls == 'vit':
                scale = vit_layer_decay_rate ** (vit_num_layers - layer_id - 1)
            else:
                scale = qllama_layer_decay_rate ** (qllama_num_layers - layer_id - 1)
            scale = min(1.0, scale)
            parameter_groups[group_name] = {
                'weight_decay': this_weight_decay,
                'params': [],
                'param_names': [],
                'lr_scale': scale,
                'group_name': group_name,
                'lr': scale * self.args.learning_rate,
            }
        parameter_groups[group_name]['params'].append(param)
        parameter_groups[group_name]['param_names'].append(name)

        rank = torch.distributed.get_rank()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    'param_names': parameter_groups[key]['param_names'],
                    'lr_scale': parameter_groups[key]['lr_scale'],
                    'lr': parameter_groups[key]['lr'],
                    'weight_decay': parameter_groups[key]['weight_decay'],
                }
            print('Param groups = %s' % json.dumps(to_display, indent=2))

    optimizer_grouped_parameters = list(parameter_groups.values())
    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

    self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    if optimizer_cls.__name__ == 'Adam8bit':
        import bitsandbytes

        manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

        skipped = 0
        for module in opt_model.modules():
            if isinstance(module, nn.Embedding):
                skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                logger.info(f'skipped {module}: {skipped / 2 ** 20}M params')
                manager.register_module_override(module, 'weight', {'optim_bits': 32})
                logger.debug(f'bitsandbytes: will optimize {module} in fp32')
        logger.info(f'skipped: {skipped / 2 ** 20}M params')

    if is_sagemaker_mp_enabled():
        import smdistributed.modelparallel.torch as smp
        self.optimizer = smp.DistributedOptimizer(self.optimizer)

    return self.optimizer


def replace_create_optimizer():
    print('Replace original create_optimizer with custom create_optimizer')
    transformers.Trainer.create_optimizer = create_optimizer
