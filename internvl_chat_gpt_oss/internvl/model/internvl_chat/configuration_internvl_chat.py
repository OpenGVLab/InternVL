# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import copy
from typing import Dict, Any, Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from .configuration_intern_vit import InternVisionConfig

logger = logging.get_logger(__name__)


class InternVLChatConfig(PretrainedConfig):
    model_type = 'internvl_chat'
    is_composition = True

    def __init__(
        self,
        vision_config: Optional[Dict[str, Any]] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        use_backbone_lora=0,
        use_llm_lora=0,
        select_layer=-1,
        force_image_size=None,
        downsample_ratio=0.5,
        template=None,
        dynamic_image_size=False,
        use_thumbnail=False,
        ps_version="v1",
        min_dynamic_patch=1,
        max_dynamic_patch=6,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {'architectures': ['InternVisionModel']}
            logger.info('vision_config is None. Initializing the InternVisionConfig with default values.')

        if llm_config is None:
            llm_config = {'architectures': ['Qwen2ForCausalLM']}
            logger.info('llm_config is None. Initializing the LlamaConfig config with default values (`LlamaConfig`).')
        assert 'architectures' in llm_config, "Should specify architecture in llm_config"

        if isinstance(vision_config, dict):
            self.vision_config = InternVisionConfig(**vision_config)
        else:
            self.vision_config = vision_config

        if isinstance(llm_config, dict):
            architecture: str = llm_config['architectures'][0]
            if architecture == 'LlamaForCausalLM':
                from transformers import LlamaConfig
                self.llm_config = LlamaConfig(**llm_config)
            elif architecture == 'Qwen2ForCausalLM':
                from transformers import Qwen2Config
                self.llm_config = Qwen2Config(**llm_config)
            elif architecture == 'Qwen3MoeForCausalLM':
                from transformers import Qwen3MoeConfig
                self.llm_config = Qwen3MoeConfig(**llm_config)
            elif architecture == 'Qwen3ForCausalLM':
                from transformers import Qwen3Config
                self.llm_config = Qwen3Config(**llm_config)
            elif architecture == 'GptOssForCausalLM':
                from transformers import GptOssConfig
                self.llm_config = GptOssConfig(**llm_config)
            else:
                raise ValueError('Unsupported architecture: {}'.format(architecture))
        else:
            self.llm_config = llm_config

        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.select_layer = select_layer
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.ps_version = ps_version  # pixel shuffle version
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.tie_word_embeddings = self.llm_config.tie_word_embeddings

        logger.info(f'vision_select_layer: {self.select_layer}')
        logger.info(f'ps_version: {self.ps_version}')
        logger.info(f'min_dynamic_patch: {self.min_dynamic_patch}')
        logger.info(f'max_dynamic_patch: {self.max_dynamic_patch}')

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
        output['select_layer'] = self.select_layer
        output['force_image_size'] = self.force_image_size
        output['downsample_ratio'] = self.downsample_ratio
        output['template'] = self.template
        output['dynamic_image_size'] = self.dynamic_image_size
        output['use_thumbnail'] = self.use_thumbnail
        output['ps_version'] = self.ps_version
        output['min_dynamic_patch'] = self.min_dynamic_patch
        output['max_dynamic_patch'] = self.max_dynamic_patch

        return output
