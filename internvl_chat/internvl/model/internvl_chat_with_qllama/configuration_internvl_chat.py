# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import copy

from transformers import LlamaConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from .configuration_internvl import InternVLConfig

logger = logging.get_logger(__name__)


class InternVLChatConfig(PretrainedConfig):
    model_type = 'internvl_chat'
    is_composition = True

    def __init__(
            self,
            internvl_config=None,
            llm_config=None,
            use_llm_lora=0,
            pad2square=False,
            select_layer=-4,
            **kwargs):
        super().__init__(**kwargs)

        if internvl_config is None:
            internvl_config = {}
            logger.info('internvl_config is None. Initializing the InternVLConfig with default values.')

        if llm_config is None:
            llm_config = {}
            logger.info('llm_config is None. Initializing the LlamaConfig config with default values (`LlamaConfig`).')

        self.internvl_config = InternVLConfig(**internvl_config)
        self.llm_config = LlamaConfig(**llm_config)
        self.num_query_token = self.internvl_config.num_query_token
        self.use_llm_lora = use_llm_lora
        self.pad2square = pad2square
        self.select_layer = select_layer

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output['internvl_config'] = self.internvl_config.to_dict()
        output['llm_config'] = self.llm_config.to_dict()
        output['model_type'] = self.__class__.model_type
        output['use_llm_lora'] = self.use_llm_lora
        output['pad2square'] = self.pad2square
        output['select_layer'] = self.select_layer
        output['template'] = self.template

        return output
