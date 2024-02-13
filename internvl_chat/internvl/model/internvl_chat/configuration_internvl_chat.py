# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import copy

from transformers import LlamaConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from .configuration_intern_vit import InternVisionConfig

logger = logging.get_logger(__name__)


class InternVLChatConfig(PretrainedConfig):
    model_type = 'internvl_chat'
    is_composition = True

    def __init__(
            self,
            vision_config=None,
            llm_config=None,
            use_backbone_lora=0,
            use_llm_lora=0,
            pad2square=False,
            select_layer=-4,
            force_image_size=None,
            downsample_ratio=0.5,
            template=None,
            **kwargs):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {}
            logger.info('vision_config is None. Initializing the InternVisionConfig with default values.')

        if llm_config is None:
            llm_config = {}
            logger.info('llm_config is None. Initializing the LlamaConfig config with default values (`LlamaConfig`).')

        self.vision_config = InternVisionConfig(**vision_config)
        self.llm_config = LlamaConfig(**llm_config)
        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.pad2square = pad2square
        self.select_layer = select_layer
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template

        logger.info(f'vision_select_layer: {self.select_layer}')

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output['vision_config'] = self.vision_config.to_dict()
        output['llm_config'] = self.llm_config.to_dict()
        output['model_type'] = self.__class__.model_type
        output['use_backbone_lora'] = self.use_backbone_lora
        output['use_llm_lora'] = self.use_llm_lora
        output['pad2square'] = self.pad2square
        output['select_layer'] = self.select_layer
        output['force_image_size'] = self.force_image_size
        output['downsample_ratio'] = self.downsample_ratio
        output['template'] = self.template

        return output
