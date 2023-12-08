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


class InternVLConfig(PretrainedConfig):
    r"""
    [`InternVLConfig`] is the configuration class to store the configuration of a
    [`InternVLModel`]. It is used to instantiate a InternVLModel according to the specified
    arguments, defining the InternViT-6B and QLLaMA configs. Instantiating a configuration with
    the defaults will yield a similar configuration to that of the InternVL architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`InternVisionConfig`].
        qllama_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`LLaMAConfig`].
        clip_embed_dim (`int`, *optional*, defaults to 768):
            Size of the embeddings from the CLIP model.
        attn_pool_num_heads (`int`, *optional*, defaults to 16):
            Number of attention heads used in the attention pooling layers.
        num_query_token (`int`, *optional*, defaults to 96):
            Number of query tokens used in the transformer.
        label_smoothing (`float`, *optional*, defaults to 0.0):
            The amount of label smoothing to apply.
        cross_attention_frequency (`int`, *optional*, defaults to 2):
            The frequency of cross-attention layers in the model.
        use_backbone_lora (`int`, *optional*, defaults to 0):
            If non-zero, indicates the use of LoRA in the backbone of the model.
        use_qllama_lora (`int`, *optional*, defaults to 0):
            If non-zero, indicates the use of LoRA in the QLLaMA of the model.
        force_image_size (`int` or `None`, *optional*):
            If not None, forces the model to use this specific image size.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        kwargs (*optional*):
            Dictionary of additional keyword arguments.
    """

    model_type = 'internvl'
    is_composition = True

    def __init__(
            self,
            vision_config=None,
            qllama_config=None,
            clip_embed_dim=768,
            attn_pool_num_heads=16,
            num_query_token=96,
            label_smoothing=0.0,
            cross_attention_frequency=2,
            use_backbone_lora=0,
            use_qllama_lora=0,
            force_image_size=None,
            initializer_range=0.02,
            **kwargs):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {}
            logger.info('vision_config is None. initializing the InternVisionConfig with default values.')

        if qllama_config is None:
            qllama_config = {}
            logger.info(
                'qllama_config is None. Initializing the InternTextConfig config with default values (`LlamaConfig`).')

        self.vision_config = InternVisionConfig(**vision_config)
        self.qllama_config = LlamaConfig(**qllama_config)
        self.qllama_config.num_query_token = num_query_token
        self.qllama_config.cross_attention_frequency = cross_attention_frequency
        self.hidden_size = self.qllama_config.hidden_size

        self.clip_embed_dim = clip_embed_dim
        self.attn_pool_num_heads = attn_pool_num_heads
        self.num_query_token = num_query_token
        self.label_smoothing = label_smoothing
        self.use_backbone_lora = use_backbone_lora
        self.use_qllama_lora = use_qllama_lora
        self.force_image_size = force_image_size
        self.initializer_range = initializer_range

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output['vision_config'] = self.vision_config.to_dict()
        output['qllama_config'] = self.qllama_config.to_dict()
        output['model_type'] = self.__class__.model_type
        return output
