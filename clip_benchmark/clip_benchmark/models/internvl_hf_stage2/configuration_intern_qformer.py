""" InternVL model configuration"""

import copy
import os
from typing import Union

from transformers import LlamaConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.modeling_auto import \
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import logging

logger = logging.get_logger(__name__)


class InternVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`InternVisionModel`]. It is used to
    instantiate a Intern vision encoder according to the specified arguments, defining the model architecture.
    Instantiating a configuration defaults will yield a similar configuration to that of the Intern architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 3200):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 12800):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 48):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 25):
            Number of attention heads for each attention layer in the Transformer encoder.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 14):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"gelu"` are supported. layer_norm_eps (`float`, *optional*, defaults
            to 1e-5): The epsilon used by the layer normalization layers.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float``, *optional*, defaults to 0.1):
            A factor for layer scale
        qkv_bias (`bool`, *optional*, defaults to `False`):
            Whether to add a bias to the queries and values in the self-attention layers.
    """

    model_type = 'intern_vision_model'

    def __init__(
            self,
            num_channels=3,
            patch_size=14,
            image_size=224,
            qkv_bias=False,
            hidden_size=3200,
            num_attention_heads=25,
            intermediate_size=12800,
            qk_normalization=True,
            num_hidden_layers=48,
            use_flash_attn=True,
            hidden_act='gelu',
            layer_norm_eps=1e-6,
            dropout=0.0,
            drop_path_rate=0.0,
            attention_dropout=0.0,
            initializer_range=0.02,  # for weight initialization
            initializer_factor=0.1,  # for layer scale
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.drop_path_rate = drop_path_rate
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.qkv_bias = qkv_bias
        self.qk_normalization = qk_normalization
        self.use_flash_attn = use_flash_attn

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> 'PretrainedConfig':
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from InternCLIPConfig
        if config_dict.get('model_type') == 'intern_vision_model':
            if 'vision_config' in config_dict:
                config_dict = config_dict['vision_config']

        if 'model_type' in config_dict and hasattr(cls, 'model_type') and config_dict['model_type'] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f'{cls.model_type}. This is not supported for all configurations of models and can yield errors.'
            )

        return cls.from_dict(config_dict, **kwargs)


class InternQformerConfig(PretrainedConfig):
    r"""
    [`InternCLIPConfig`] is the configuration class to store the configuration of a
    [`InternCLIPModel`]. It is used to instantiate a InternVL model according to the specified
    arguments, defining the vision model and language model configs. Instantiating a configuration with
    the defaults will yield a similar configuration to that of the InternVL architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`InternVisionConfig`].
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize any [`PretrainedConfig`].
        clip_embed_dim (`int`, *optional*, defaults to 768):
            Size of the embeddings from the CLIP model.

        kwargs (*optional*):
            Dictionary of keyword arguments.
    """

    model_type = 'intern_qformer'
    is_composition = True

    def __init__(
            self,
            vision_config=None,
            text_config=None,
            clip_embed_dim=768,
            attn_pool_num_heads=16,
            num_query_token=96,
            max_txt_len=32,
            label_smoothing=0.0,
            cross_attention_frequency=2,
            use_backbone_lora=0,
            use_qformer_lora=0,
            force_image_size=None,
            **kwargs):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {}
            logger.info('vision_config is None. initializing the InternVisionConfig with default values.')

        if text_config is None:
            text_config = {}
            logger.info(
                'text_config is None. Initializing the InternTextConfig config with default values (`LlamaConfig`).')

        self.vision_config = InternVisionConfig(**vision_config)
        self.text_config = LlamaConfig(**text_config)
        self.text_config.num_query_token = num_query_token
        self.text_config.cross_attention_frequency = cross_attention_frequency
        self.hidden_size = self.text_config.hidden_size
        self.tie_word_embeddings = self.text_config.tie_word_embeddings
        self.is_encoder_decoder = self.text_config.is_encoder_decoder

        self.clip_embed_dim = clip_embed_dim
        self.attn_pool_num_heads = attn_pool_num_heads
        self.num_query_token = num_query_token
        self.max_txt_len = max_txt_len
        self.label_smoothing = label_smoothing
        self.use_decoder_only_language_model = self.text_config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        self.initializer_range = 0.02
        self.use_backbone_lora = use_backbone_lora
        self.use_qformer_lora = use_qformer_lora
        self.force_image_size = force_image_size

    @classmethod
    def from_vision_text_configs(
            cls,
            vision_config: InternVisionConfig,
            text_config: PretrainedConfig,
            **kwargs,
    ):
        r"""
        Instantiate a [`InternCLIPConfig`] (or a derived class) from a Intern vision model and
        language model configurations.

        Returns:
            [`InternCLIPConfig`]: An instance of a configuration object
        """

        return cls(
            vision_config=vision_config.to_dict(),
            text_config=text_config.to_dict(),
            **kwargs,
        )

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output['vision_config'] = self.vision_config.to_dict()
        output['text_config'] = self.text_config.to_dict()
        output['model_type'] = self.__class__.model_type
        return output


if __name__ == '__main__':
    config = InternQformerConfig.from_pretrained('./intern_clip_13b')
    print(config)
