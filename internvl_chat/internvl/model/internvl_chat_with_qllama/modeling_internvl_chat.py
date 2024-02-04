# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch.utils.checkpoint
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from .configuration_internvl_chat import InternVLChatConfig
from .modeling_internvl import InternVLModel

logger = logging.get_logger(__name__)


class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    _no_split_modules = ['InternVisionEncoderLayer', 'LlamaDecoderLayer', 'LlamaForCausalLM']

    def __init__(self, config: InternVLChatConfig, internvl=None, language_model=None):
        super().__init__(config)

        num_query_token = config.internvl_config.num_query_token
        image_size = config.internvl_config.force_image_size or config.internvl_config.vision_config.image_size
        patch_size = config.internvl_config.vision_config.patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = (image_size // patch_size) ** 2
        self.num_query_token = num_query_token
        print('num_image_token:', self.num_image_token)
        print('num_query_token:', self.num_query_token)
        if internvl is not None:
            self.internvl = internvl
        else:
            self.internvl = InternVLModel(config.internvl_config)  # frozen
        if language_model is not None:
            self.language_model = language_model
        else:
            self.language_model = LlamaForCausalLM(config.llm_config)  # frozen
        vit_hidden_size = config.internvl_config.vision_config.hidden_size
        qllama_hidden_size = config.internvl_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size),
            nn.Linear(vit_hidden_size, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.mlp2 = nn.Sequential(
            nn.LayerNorm(qllama_hidden_size),
            nn.Linear(qllama_hidden_size, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.img_context_token_id = None
        self.query_context_token_id = None
        self.internvl_tokenizer = None
        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora)
            self.use_llm_lora = True
        else:
            self.use_llm_lora = False

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                            'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM'
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.print_trainable_parameters()

    def forward(
            self,
            pixel_values: torch.FloatTensor = None,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            question_input_ids: Optional[torch.LongTensor] = None,
            question_attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            vit_embeds, qllama_embeds = self.extract_feature(
                pixel_values, question_input_ids, question_attention_mask)

            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            temp_embeds = torch.zeros_like(input_embeds)
            temp_embeds[selected] = vit_embeds.reshape(-1, C)
            selected_bf16 = selected.to(input_embeds.dtype).unsqueeze(-1)
            input_embeds = input_embeds * (1 - selected_bf16) + temp_embeds * selected_bf16
            # input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)

            selected = (input_ids == self.query_context_token_id)
            length = selected.sum()
            temp_embeds = torch.zeros_like(input_embeds)
            temp_embeds[selected] = qllama_embeds.reshape(-1, C)[:length]
            selected_bf16 = selected.to(input_embeds.dtype).unsqueeze(-1)
            input_embeds = input_embeds * (1 - selected_bf16) + temp_embeds * selected_bf16
            # input_embeds[selected] = input_embeds[selected] * 0.0 + qllama_embeds.reshape(-1, C)[:length]
            input_embeds = input_embeds.reshape(B, N, C)
        else:
            print('pure text forward')

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def extract_feature(self, pixel_values, question_input_ids, question_attention_mask):
        vit_embeds, qllama_embeds = self.internvl.get_image_features(
            pixel_values, question_input_ids, question_attention_mask, self.select_layer)
        vit_embeds = self.mlp1(vit_embeds[:, 1:, :])
        qllama_embeds = self.mlp2(qllama_embeds)
        return vit_embeds, qllama_embeds

    def chat(self, tokenizer, pixel_values, question, generation_config,
             IMG_START_TOKEN='<IMG>', IMG_END_TOKEN='</IMG>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             QUERY_START_TOKEN='<QUERY>', QUERY_END_TOKEN='</QUERY>', QUERY_CONTEXT_TOKEN='<QUERY_CONTEXT>',
             internvl_tokenizer_path='/mnt/petrelfs/wangwenhai/workspace/InternVL-release/internvl_chat/data/llm/internvl_14b_224px'):

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        query_context_token_id = tokenizer.convert_tokens_to_ids(QUERY_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id
        self.query_context_token_id = query_context_token_id

        from internvl.conversation import get_conv_template

        template = get_conv_template(self.template)
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token + IMG_END_TOKEN
        query_tokens = QUERY_START_TOKEN + QUERY_CONTEXT_TOKEN * self.num_query_token + QUERY_END_TOKEN
        template.append_message(template.roles[0], image_tokens + query_tokens + '\n' + question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()

        if self.internvl_tokenizer is None:
            self.internvl_tokenizer = LlamaTokenizer.from_pretrained(
                internvl_tokenizer_path, add_eos_token=True)
        tokenized_questions = self.internvl_tokenizer(question, return_tensors='pt')
        question_input_ids = tokenized_questions['input_ids'].cuda().unsqueeze(0)
        question_attention_mask = tokenized_questions['attention_mask'].cuda().unsqueeze(0)

        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            question_input_ids=question_input_ids,
            question_attention_mask=question_attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        query_to_print = query.replace(image_tokens, '<image>').replace(query_tokens, '<query>')
        print(query_to_print, response)
        return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.FloatTensor,
            attention_mask: torch.LongTensor,
            question_input_ids: torch.FloatTensor,
            question_attention_mask: torch.LongTensor,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        assert self.query_context_token_id is not None
        if visual_features is not None:
            vit_embeds, qllama_embeds = visual_features
        else:
            vit_embeds, qllama_embeds = self.extract_feature(
                pixel_values, question_input_ids, question_attention_mask)

        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        assert selected.sum() != 0
        input_embeds[selected] = vit_embeds.reshape(-1, C)

        selected = (input_ids == self.query_context_token_id)
        assert selected.sum() != 0
        input_embeds[selected] = qllama_embeds.reshape(-1, C)

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs
