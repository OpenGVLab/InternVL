# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Mostly copy-paste from BEiT library:
https://github.com/microsoft/unilm/blob/master/beit/semantic_segmentation/mmcv_custom/layer_decay_optimizer_constructor.py
"""

import json

from mmcv.runner import (OPTIMIZER_BUILDERS, DefaultOptimizerConstructor,
                         get_dist_info)
from mmseg.utils import get_root_logger


def get_num_layer_for_vit(var_name, num_max_layer):
    var_name = var_name.replace('cb_modules.0.', '').replace('cb_modules.1.', '')
    var_name = var_name.replace('levels.', 'layers.')
    if var_name in ('backbone.cls_token', 'backbone.mask_token',
                    'backbone.pos_embed', 'backbone.visual_embed'):
        return 0
    elif var_name.startswith('backbone.visual_embed'):
        return 0
    elif var_name.startswith('backbone.patch_embed'):
        return 0
    elif var_name.startswith('backbone.blocks') or var_name.startswith('backbone.layers'):
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    else:
        return num_max_layer - 1


@OPTIMIZER_BUILDERS.register_module()
class CustomLayerDecayOptimizerConstructor(DefaultOptimizerConstructor):
    def add_params(self, params, module, prefix='', is_dcn_module=None):
        """Add all parameters of module to the params list.
        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.
        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module
            is_dcn_module (int|float|None): If the current module is a
                submodule of DCN, `is_dcn_module` will be passed to
                control conv_offset layer's learning rate. Defaults to None.
        """
        parameter_groups = {}
        logger = get_root_logger()
        logger.info(self.paramwise_cfg)
        num_layers = self.paramwise_cfg.get('num_layers') + 2
        layer_decay_rate = self.paramwise_cfg.get('layer_decay_rate')
        logger.info('Build LayerDecayOptimizerConstructor %f - %d' % (layer_decay_rate, num_layers))
        weight_decay = self.base_wd

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith('.bias') or name in ('pos_embed', 'cls_token'):
                group_name = 'no_decay'
                this_weight_decay = 0.
            else:
                group_name = 'decay'
                this_weight_decay = weight_decay

            layer_id = get_num_layer_for_vit(name, num_layers)
            group_name = 'layer_%d_%s' % (layer_id, group_name)

            if group_name not in parameter_groups:
                scale = layer_decay_rate ** (num_layers - layer_id - 1)

                parameter_groups[group_name] = {
                    'weight_decay': this_weight_decay,
                    'params': [],
                    'param_names': [],
                    'lr_scale': scale,
                    'group_name': group_name,
                    'lr': scale * self.base_lr,
                }

            parameter_groups[group_name]['params'].append(param)
            parameter_groups[group_name]['param_names'].append(name)
        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    'param_names': parameter_groups[key]['param_names'],
                    'lr_scale': parameter_groups[key]['lr_scale'],
                    'lr': parameter_groups[key]['lr'],
                    'weight_decay': parameter_groups[key]['weight_decay'],
                }
            logger.info('Param groups = %s' % json.dumps(to_display, indent=2))

        params.extend(parameter_groups.values())
