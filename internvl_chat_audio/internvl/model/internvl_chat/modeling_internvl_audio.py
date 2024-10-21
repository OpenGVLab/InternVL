# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import warnings
from typing import Any, List, Optional, Tuple, Union
from PIL import Image, ImageDraw
from io import BytesIO
import requests
import torch.distributed as dist
import torch.utils.checkpoint
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from internvl.model.phi3.modeling_phi3 import Phi3ForCausalLM
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging
from internvl.conversation import get_conv_template
from .configuration_internvl_chat import InternVLChatConfig
from .modeling_intern_vit import InternVisionModel
from .modeling_internvl_chat import InternVLChatModel
from .configuration_internvl_audio_chat import InternVLChatAudioConfig
from internvl.model.audio.modeling_whisper import AudioWhisperModel
from internvl.conversation import get_conv_template
from internvl.train.dataset_audio import *
from internvl.train.dataset import dynamic_preprocess
from internvl.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                      IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN, QUAD_END_TOKEN,
                                      QUAD_START_TOKEN, REF_END_TOKEN,
                                      REF_START_TOKEN, AUDIO_START_TOKEN,
                                      AUDIO_END_TOKEN, AUDIO_CONTEXT_TOKEN)



def load_audio(audio_file, audio_processor):
    audio_values, _ = librosa.load(audio_file, sr=16000) # sample rate should be 16000
    
    audio_process_values = audio_processor(audio_values, sampling_rate=16000, return_tensors="pt")
    input_features = audio_process_values['input_features']
    audio_len_after_cnn = audio_process_values['audio_len_after_cnn']
    audio_token_num = audio_process_values['audio_token_num']
                

    audio_input = {'audio_values': input_features,
                   'audio_len_after_cnn': audio_len_after_cnn,
                   'audio_token_num': audio_token_num,
                   }
    return audio_input


class InternVLChatAudioModel(InternVLChatModel):

    def __init__(self, config: InternVLChatAudioConfig, vision_model=None, language_model=None, audio_model=None):
        super().__init__(config, vision_model, language_model)
        if audio_model is not None:
            self.audio_model = audio_model
        else:
            self.audio_model = AudioWhisperModel(config.audio_config)

        audio_hidden_size = config.audio_config.d_model
        llm_hidden_size = config.llm_config.hidden_size
        self.mlp2 = nn.Sequential(
                nn.LayerNorm(audio_hidden_size),
                nn.Linear(audio_hidden_size, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size)
            )   # mlp2: audio feature mapping

        self.audio_context_token_id = None

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def extract_audio_feature(self, audio_values, audio_len_after_cnn):

        audio_values = audio_values.squeeze(1)

        #TODO: construct audio padding_mask in loader
        max_len_in_batch = int(torch.max(audio_len_after_cnn).item())

        padding_mask = torch.ones([audio_values.size(0), max_len_in_batch]).to(dtype=audio_values.dtype,
                                                                               device=audio_values.device)
        for index in range(len(audio_values)):
            padding_mask[index, :int(audio_len_after_cnn[index].item())] = 0

        last_hidden_state = self.audio_model(audio_values, padding_mask, audio_len_after_cnn)  # (bs, max_token_num, 1280)

        audio_embeds = self.mlp2(last_hidden_state)

        return audio_embeds


    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            audio_values: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            audio_flags: Optional[torch.LongTensor] = None,
            audio_len_after_cnn: Optional[torch.LongTensor] = None,
            audio_token_num: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            statistics: Optional[torch.LongTensor] = None,
            loss_weight: Optional[List] = None,
            loss_reduction_all_gather: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')
            if statistics is not None:
                num_samples, num_padding_tokens, num_padding_images = statistics.tolist()
                self.num_samples += num_samples
                print(f'total_samples={self.num_samples}, {num_samples=}, {num_padding_tokens=}, {num_padding_images=}')


        input_ids = input_ids.reshape(B * N)
        img_selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[img_selected] = input_embeds[img_selected] * 0.0 + vit_embeds.reshape(-1, C)
            ignore_flag = False
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[img_selected].shape={input_embeds[img_selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = img_selected.sum()
            input_embeds[img_selected] = input_embeds[img_selected] * 0.0 + vit_embeds[:n_token]
            ignore_flag = True



        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            audio_batch_size = audio_values.shape[0]
            print(f'audio batch size: {audio_batch_size}, audios per sample: {audio_batch_size / B}')

        audio_embeds = self.extract_audio_feature(audio_values, audio_len_after_cnn)  # (audio_num, n_frame, C)

        output_audios = []
        for i in range(len(audio_token_num)):
            if audio_flags[i] > 0:
                token_num = int(audio_token_num[i].item())
                audio = audio_embeds[i][:token_num]   # 提取有效的token
                output_audios.append(audio)

        if len(output_audios):
            output_audios = torch.cat(output_audios, dim=0)
            audio_selected = (input_ids == self.audio_context_token_id)
            input_embeds[audio_selected] = input_embeds[audio_selected] * 0.0 + output_audios.reshape(-1, C)

        # # optional 2
        # valid_audio_mask = torch.zeros(audio_embeds.shape[:2], dtype=torch.bool)
        # for i in range(len(audio_token_num)):
        #     token_num = int(audio_token_num[i].item())
        #     valid_audio_mask[i, :token_num] = True
        # output_audios_2 = audio_embeds[valid_audio_mask]


        input_embeds = input_embeds.reshape(B, N, C)

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
        if labels is not None and loss_weight is not None:
            loss_weight = torch.tensor(loss_weight,
                                       dtype=torch.float32,
                                       device=labels.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = loss_weight[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            shift_logits = shift_logits.view(
                -1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_weights = shift_weights.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            shift_weights = shift_weights.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            shift_weights_sum = shift_weights.sum()
            if loss_reduction_all_gather:
                dist.all_reduce(shift_weights_sum, op=dist.ReduceOp.AVG)

            loss = loss * shift_weights
            loss = loss.sum() / shift_weights_sum
            if ignore_flag:
                loss = loss * 0.0
        elif labels is not None:
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
            if ignore_flag:
                loss = loss * 0.0

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



    @torch.no_grad()
    def generate(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.FloatTensor,
            attention_mask: torch.LongTensor,
            visual_features: Optional[torch.FloatTensor] = None,
            audio_values: Optional[torch.FloatTensor] = None,
            audio_len_after_cnn: Optional[bool] = None,
            audio_token_num: Optional[bool] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        assert self.audio_context_token_id is not None

        vit_embeds = None
        if visual_features is not None:
            vit_embeds = visual_features
        elif pixel_values is not None:
            vit_embeds = self.extract_feature(pixel_values)

        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)

        if vit_embeds is not None:
            selected = (input_ids == self.img_context_token_id)
            input_embeds[selected] = vit_embeds.reshape(-1, C)

        if audio_values is not None and audio_len_after_cnn is not None and audio_token_num is not None:
            audio_embeds = self.extract_audio_feature(audio_values, audio_len_after_cnn)
            output_audios = []
            for i in range(len(audio_token_num)):
                token_num = int(audio_token_num[i].item())
                audio = audio_embeds[i][:token_num]
                output_audios.append(audio)
            output_audios = torch.cat(output_audios, dim=0)
            selected = (input_ids == self.audio_context_token_id)
            input_embeds[selected] = output_audios.reshape(-1, C)

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
    

class Chat:
    def __init__(
            self,
            model,
            tokenizer,
            audio_processor,
            device,
            temperature=0.7,
            top_p=0.5,
            top_k=20,
            repetition_penalty=1.1,
            max_new_tokens=1024,
            conv_style="internlm2-chat",
            do_sample=True,
            image_size=448,
            dynamic_image_size=False,
            scale_threshold='old',
            min_dynamic_patch=1,
            max_dynamic_patch=12,
            use_thumbnail=True
    ):
        self.conv_style = conv_style
        self.model = model
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor

        self.device = device
        self.dtype = model.dtype

        self.eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(["<|im_end|>"])[0]]

        self.conv = get_conv_template(conv_style)
        self.image_size = model.config.force_image_size or model.config.vision_config.image_size
        self.dynamic_image_size = dynamic_image_size
        self.scale_threshold = scale_threshold
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.use_thumbnail = model.config.use_thumbnail

        self.max_new_tokens = max_new_tokens
        self.do_sample =  do_sample
        self.streamer = None
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty

    def load_image(self, image_file):
        if 's3://' in image_file:
            image = Image.open(io.BytesIO(client.get(image_file)))
        else:
            if image_file.startswith('http') or image_file.startswith('https'):
                response = requests.get(image_file)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                image = Image.open(image_file).convert('RGB')
        
        transform = build_transform(is_train=False, input_size=self.image_size)

        if self.dynamic_image_size:
            if self.scale_threshold == "old":
                images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                        image_size=self.image_size, use_thumbnail=self.use_thumbnail)
            else:
                images = dynamic_preprocess(image, data_item=None, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                        image_size=self.image_size, use_thumbnail=self.use_thumbnail,
                                        normalize_type=self.normalize_type, scale_threshold=float(self.scale_threshold)) #not detection data
                # raise NotImplementedError
        else:
            images = [image]
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)
        return pixel_values

    def ask(self, text, conv, modal_type="audio"):
        assert modal_type in ["text", "image", "audio", "image_audio"]
        conversations = []

        if len(conv.messages) > 0 or modal_type == "text":
            conv.append_message(conv.roles[0], text)
        elif modal_type == "image":
            conv.append_message(conv.roles[0], "<image>" + "\n" + text)
        elif modal_type == "audio":
            if text is None:
                conv.append_message(conv.roles[0], "<audio>" + "\n")
            else:
                conv.append_message(conv.roles[0], "<audio>" + "\n" + text)
        elif modal_type == "image_audio":
            if text is None:
                conv.append_message(conv.roles[0], "<image>\n<audio>\n" )
            else:
                conv.append_message(conv.roles[0], "<image>\n<audio>\n" + text)
        else:
            raise ValueError

        conv.append_message(conv.roles[1], None)
        conversations.append(conv.get_prompt())
        return conversations

    @torch.no_grad()
    def answer(self, conversations, img_path=None, audio_path=None, modal_type='audio'):
        pixel_values = None
        audio_values = None
        audio_is_longer = None
        audio_token_num = 0
        audio_len_after_cnn = 0
        
        if img_path:
            pixel_values = self.load_image(img_path)
            pixel_values = pixel_values.to(self.device, dtype=self.dtype)

            num_image_token = pixel_values.shape[0]*256
            image_tokens = f"{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token}{IMG_END_TOKEN}" 
            new_conversations = []
            for conversation in conversations:
                conversation = conversation.replace("<image>", image_tokens)
                new_conversations.append(conversation)
            conversations = new_conversations
        
        if audio_path:
            audio_input = load_audio(audio_path, self.audio_processor)
            audio_values = audio_input['audio_values'].to(self.device, dtype=self.dtype)
            audio_len_after_cnn = audio_input['audio_len_after_cnn']
            audio_token_num = audio_input['audio_token_num']

            audio_tokens = f"{AUDIO_START_TOKEN}{AUDIO_CONTEXT_TOKEN * audio_token_num}{AUDIO_END_TOKEN}"
            new_conversations = []
            for conversation in conversations:
                conversation = conversation.replace("<audio>", audio_tokens)
                new_conversations.append(conversation)
            conversations = new_conversations
        audio_token_num = torch.tensor([audio_token_num])
        audio_len_after_cnn = torch.tensor([audio_len_after_cnn])

        model_inputs = self.tokenizer(
            conversations,
            return_tensors="pt",
        )
        model_inputs.pop("token_type_ids", None)
        input_ids = model_inputs["input_ids"].to(self.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=self.device)
        
        if img_path is None and audio_path is None:
            outputs = self.model.language_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                eos_token_id=self.eos_token_id,
            )
        else:
            outputs = self.model.generate(
                pixel_values=pixel_values,
                audio_values=audio_values,
                audio_len_after_cnn=audio_len_after_cnn,
                audio_token_num=audio_token_num,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                eos_token_id=self.eos_token_id,
            )
        outputs = outputs[0].cpu().tolist()
        response = self.tokenizer.decode(outputs, skip_special_tokens=True)
        response = response.split('assistant\n')[-1].split('<|im_end|>')[0].strip()  # 纯llm chat会输出所有的内容: prompt+answer

        return response
    
    @torch.no_grad()
    def answer_batch(self, conversations, img_paths=None, audio_paths=None):
        pixel_values = None
        audio_values = None
        audio_is_longer = None
        _pad_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = 'left'
        
        if img_paths:
            tensors = []
            nums_images = []
            for i, image_file in enumerate(img_paths):
                tensor = self.load_image(image_file)
                tensor = tensor.to(self.device, dtype=self.dtype)
                tensors.append(tensor)
                nums_images.append(tensor.shape[0])
            pixel_values = torch.concat(tensors, dim=0)
            pixel_values = pixel_values.to(self.device, dtype=self.dtype)

            new_conversations = []
            for i, conversation in enumerate(conversations):
                num_image_token = nums_images[i]*256
                image_tokens = f"{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token}{IMG_END_TOKEN}" 
                conversation = conversation.replace("<image>", image_tokens)
                new_conversations.append(conversation)
            conversations = new_conversations
        if audio_paths:
            audio_values = []
            audio_len_after_cnns = []
            audio_token_nums = []

            for i, audio_file in enumerate(audio_paths):
                audio_input = load_audio(audio_file, self.audio_processor)
                audio_value = audio_input['audio_values'].to(self.device, dtype=self.dtype)
                audio_len_after_cnn = audio_input['audio_len_after_cnn']
                audio_token_num = audio_input['audio_token_num']
                audio_values.append(audio_value)
                audio_len_after_cnns.append(audio_len_after_cnn)
                audio_token_nums.append(audio_token_num)
            
            new_conversations = []
            for i, conversation in enumerate(conversations):
                audio_token_num = audio_token_nums[i]
                audio_tokens = f"{AUDIO_START_TOKEN}{AUDIO_CONTEXT_TOKEN * audio_token_num}{AUDIO_END_TOKEN}"
                conversation = conversation.replace("<audio>", audio_tokens)
                new_conversations.append(conversation)
            conversations = new_conversations
            
            audio_token_num = torch.tensor(audio_token_nums)
            audio_len_after_cnn = torch.tensor(audio_len_after_cnns)
            audio_values = torch.concat(audio_values, dim=0)
            audio_values = audio_values.to(self.device, dtype=self.dtype)

      
        model_inputs = self.tokenizer(
            conversations,
            return_tensors="pt",
            padding='longest'
        )
        model_inputs.pop("token_type_ids", None)
        input_ids = model_inputs["input_ids"].to(self.device)
        # attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=self.device)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        
        if img_paths is None and audio_paths is None:
            outputs = self.model.language_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                eos_token_id=self.eos_token_id,
            )
        else:
            outputs = self.model.generate(
                pixel_values=pixel_values,
                audio_values=audio_values,
                audio_len_after_cnn=audio_len_after_cnn,
                audio_token_num=audio_token_num,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                eos_token_id=self.eos_token_id,
            )
        responses = []
        for i in range(len(outputs)):
            cur_outputs = outputs[i].cpu().tolist()
        
            response = self.tokenizer.decode(cur_outputs, skip_special_tokens=True)
            responses.append(response.strip())
        # response = response.split('assistant\n')[-1].split('<|im_end|>')[0].strip()  # 纯llm chat会输出所有的内容: prompt+answer

        self.tokenizer.padding_side = _pad_side
        return responses 

def single_qa(chat, question=None, img_path=None, audio_path=None, vis_width=336, system_message=None):
    chat.conv = get_conv_template("internlm2-chat").copy()
    if system_message:
        chat.conv.system_message = system_message
    if img_path is not None and audio_path is None:
        modal_type = 'image'
    elif img_path is None and audio_path is not None:
        modal_type = 'audio' 
    elif img_path is not None and audio_path is not None:
        modal_type = "image_audio"
    else:
        modal_type = 'text' 
    # print('modal type:', modal_type)
    conversations = chat.ask(text=question, conv=chat.conv, modal_type=modal_type)
    # print('conversations:', conversations)
    outputs = chat.answer(conversations, img_path=img_path, audio_path=audio_path, modal_type=modal_type)
    # NOTE: strip is important to align with the training data.
    chat.conv.messages[-1][1] = outputs.strip()
    answer = outputs.strip()
    # print('Q:', question)
    # print('A:', answer)
    return answer      