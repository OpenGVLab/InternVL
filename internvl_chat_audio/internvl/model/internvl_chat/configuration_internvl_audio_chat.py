# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import copy

from internvl.model.internlm2.configuration_internlm2 import InternLM2Config
from internvl.model.phi3.configuration_phi3 import Phi3Config
from transformers import AutoConfig, LlamaConfig, Qwen2Config
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from internvl.model.audio.configuration_whisper import WhisperConfig

from .configuration_intern_vit import InternVisionConfig
from .configuration_internvl_chat import InternVLChatConfig

logger = logging.get_logger(__name__)




class InternVLChatAudioConfig(InternVLChatConfig):
    model_type = "internvl_chat"
    is_composition = True
    
    def __init__(
            self,
            vision_config=None,
            audio_config=None,
            llm_config=None,
            **kwargs):
        super().__init__(vision_config, llm_config,  **kwargs)

        if audio_config is None:
            audio_config = {}
            logger.info('audio_config is None. Initializing the Audioconfig with default values.')
        self.audio_config = WhisperConfig(**audio_config)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = super().to_dict()
        output['audio_config'] = self.audio_config.to_dict()
        return output