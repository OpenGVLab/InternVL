import requests
import torch
import torchvision.transforms as transforms
from internvl_clip import load_internvl_clip
from internvl_huggingface import load_internvl_clip_huggingface
from internvl_huggingface.configuration_intern_clip import (InternCLIPConfig,
                                                            InternVisionConfig)
from internvl_huggingface.modeling_intern_clip import (InternCLIPModel,
                                                       InternVisionModel)
from PIL import Image
from torchvision.transforms import InterpolationMode
from transformers import CLIPImageProcessor, LlamaConfig, LlamaTokenizer


def check_preprocessor():
    def _convert_to_rgb(image):
        return image.convert('RGB')

    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)

    transform = transforms.Compose([
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        _convert_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    out1 = transform(image)
    processor = CLIPImageProcessor.from_pretrained('./internvl_huggingface/intern_clip_vit_6b')
    out2 = processor(images=image, return_tensors='pt')
    print('diff:', (out1 - out2['pixel_values']).mean())


def load_internvl_weights():
    print('loading Intern-6B from original checkpoint!')
    init_ckpt = '/mnt/petrelfs/share_data/wangwenhai/internvl/' \
                '6b_vit_exp126_clip_alpaca_7b_laion5b_peak_1e-5_256gpu_all_trainable_degradation.sh/' \
                '1499/mp_rank_00_model_states.pt'
    ckpt = torch.load(init_ckpt, 'cpu')
    new_ckpt = {}
    for k, v in ckpt['module'].items():
        if 'bamboo' in k or 'predictor' in k or 'decoder' in k or 'loss' in k:
            continue
        if 'target' in k:
            continue
        new_k = k.replace('clip.transformer.', 'transformer.')
        new_k = new_k.replace('clip.text_projection', 'text_projection')
        new_k = new_k.replace('clip.logit_scale', 'logit_scale')
        new_k = new_k.replace('cls_token', 'embeddings.class_embedding')
        new_k = new_k.replace('pos_embed', 'embeddings.position_embedding')
        new_k = new_k.replace('patch_embed.proj.weight', 'embeddings.patch_embedding.weight')
        new_k = new_k.replace('patch_embed.proj.bias', 'embeddings.patch_embedding.bias')
        new_k = new_k.replace('ls1.gamma', 'ls1')
        new_k = new_k.replace('ls2.gamma', 'ls2')
        new_k = new_k.replace('blocks.', 'encoder.layers.')
        if 'norm3' in new_k or 'text_projection' in new_k or 'logit_scale' in new_k:
            continue
        if 'transformer.' in new_k or 'clip_projector.' in new_k:
            continue
        if new_k == 'target_pos_embed' or new_k == 'mask_token':
            continue
        if 'grad_norm' in new_k:
            continue
        new_ckpt[new_k] = v
    return new_ckpt


def save_intern_vision_model():
    image = torch.randn(1, 3, 224, 224).to(torch.float16).cuda()
    vision_config = InternVisionConfig()
    model = InternVisionModel(vision_config).eval()

    ckpt = load_internvl_weights()
    ckpt['video_embeddings.class_embedding'] = ckpt['embeddings.class_embedding']
    video_pos = torch.cat([
        ckpt['embeddings.position_embedding'][:, :1, :],
        ckpt['embeddings.position_embedding'][:, 1:, :],
        ckpt['embeddings.position_embedding'][:, 1:, :],
        ckpt['embeddings.position_embedding'][:, 1:, :],
        ckpt['embeddings.position_embedding'][:, 1:, :]], dim=1
    )
    print(video_pos.shape)
    ckpt['video_embeddings.position_embedding'] = video_pos
    ckpt['video_embeddings.patch_embedding.weight'] = ckpt[
        'embeddings.patch_embedding.weight'].unsqueeze(2).repeat(1, 1, 2, 1, 1)
    ckpt['video_embeddings.patch_embedding.bias'] = ckpt['embeddings.patch_embedding.bias']

    message = model.load_state_dict(ckpt, strict=False)
    print(message)
    model.save_pretrained('./internvl_huggingface/intern_clip_vit_6b')


def save_intern_clip_model():
    vision_config = InternVisionConfig()
    llama_path = '/mnt/petrelfs/share_data/chenzhe1/data/llm/chinese_alpaca_lora_7b'
    text_config = LlamaConfig.from_pretrained(llama_path)
    clip_config = InternCLIPConfig(vision_config.to_dict(), text_config.to_dict())
    model = InternCLIPModel(clip_config).eval()

    def load_internvl_weights():
        print('loading Intern-6B from original checkpoint!')
        init_ckpt = '/mnt/petrelfs/share_data/wangwenhai/internvl/' \
                    '6b_vit_exp126_clip_alpaca_7b_laion5b_peak_1e-5_256gpu_all_trainable_degradation.sh/' \
                    '1499/mp_rank_00_model_states.pt'
        ckpt = torch.load(init_ckpt, 'cpu')
        new_ckpt = {}
        for k, v in ckpt['module'].items():
            if 'bamboo' in k or 'predictor' in k or 'decoder' in k or 'loss' in k:
                continue
            if 'target' in k:
                continue
            new_k = k.replace('clip.transformer.', 'language_model.base_model.model.model.')
            new_k = new_k.replace('clip.text_projection', 'text_projection')
            new_k = new_k.replace('clip.logit_scale', 'logit_scale')
            new_k = new_k.replace('cls_token', 'vision_model.embeddings.class_embedding')
            new_k = new_k.replace('pos_embed', 'vision_model.embeddings.position_embedding')
            new_k = new_k.replace('patch_embed.proj.weight', 'vision_model.embeddings.patch_embedding.weight')
            new_k = new_k.replace('patch_embed.proj.bias', 'vision_model.embeddings.patch_embedding.bias')
            new_k = new_k.replace('ls1.gamma', 'ls1')
            new_k = new_k.replace('ls2.gamma', 'ls2')
            new_k = new_k.replace('blocks.', 'vision_model.encoder.layers.')
            if 'norm3' in new_k:
                continue
            if new_k == 'target_pos_embed' or new_k == 'mask_token':
                continue
            if 'grad_norm' in new_k:
                continue
            new_ckpt[new_k] = v
        return new_ckpt

    ckpt = load_internvl_weights()
    ckpt['vision_model.video_embeddings.class_embedding'] = ckpt['vision_model.embeddings.class_embedding']
    video_pos = torch.cat([
        ckpt['vision_model.embeddings.position_embedding'][:, :1, :],
        ckpt['vision_model.embeddings.position_embedding'][:, 1:, :],
        ckpt['vision_model.embeddings.position_embedding'][:, 1:, :],
        ckpt['vision_model.embeddings.position_embedding'][:, 1:, :],
        ckpt['vision_model.embeddings.position_embedding'][:, 1:, :]], dim=1
    )
    print(video_pos.shape)
    ckpt['vision_model.video_embeddings.position_embedding'] = video_pos
    ckpt['vision_model.video_embeddings.patch_embedding.weight'] = ckpt[
        'vision_model.embeddings.patch_embedding.weight'].unsqueeze(2).repeat(1, 1, 2, 1, 1)
    ckpt['vision_model.video_embeddings.patch_embedding.bias'] = ckpt['vision_model.embeddings.patch_embedding.bias']

    message = model.load_state_dict(ckpt, strict=False)
    model.language_model.merge_and_unload()
    model.language_model = model.language_model.base_model.model
    print(message)
    model.save_pretrained('./internvl_huggingface/intern_clip_13b')
    # TODO: 把13B模型转完格式保存下来，测试retrieval点数看是否正常


def save_tokenizer():
    llama_path = '/mnt/petrelfs/share_data/chenzhe1/data/llm/chinese_alpaca_lora_7b'
    tokenizer = LlamaTokenizer.from_pretrained(llama_path)
    tokenizer.save_pretrained('./internvl_huggingface/intern_clip_vit_6b')
    tokenizer.save_pretrained('./internvl_huggingface/intern_clip_13b')


def check_huggingface_model():
    model1, transform1, tokenizer1 = load_internvl_clip_huggingface()
    model2, transform2, tokenizer2 = load_internvl_clip(model_name='exp126', pretrained='1499',
                                                        cache_dir=None, device='cpu')
    model1.eval()
    model2.eval()
    print('tokenizer1:', tokenizer1(['hello world!']))
    print('tokenizer2:', tokenizer2(['hello world!']))
    model1 = model1.half().cuda()
    model2 = model2.half().cuda()
    image = torch.rand(1, 3, 224, 224).half().cuda()
    out1 = model1.encode_image(image)
    out2 = model2.encode_image(image)
    print('image out1:', out1.shape, out1[:20], out1.mean(), out1.std())
    print('image out2:', out2.shape, out2[:20], out2.mean(), out2.std())

    input_ids1 = tokenizer1(['hello world!']).cuda()
    input_ids2 = tokenizer2(['hello world!']).cuda()
    out1 = model1.encode_text(input_ids1)
    out2 = model2.encode_text(input_ids2)
    print('text out1:', out1.shape, out1[:20], out1.mean(), out1.std())
    print('text out2:', out2.shape, out2[:20], out2.mean(), out2.std())


def check_readme():
    # check InternVisionModel
    from internvl_huggingface import InternVisionModel

    model_path = '/mnt/petrelfs/share_data/wangwenhai/internvl/internvl_huggingface/intern_clip_vit_6b'
    model = InternVisionModel.from_pretrained(model_path).half().cuda()
    pixel_values = torch.rand(1, 3, 224, 224).half().cuda()
    vision_outputs = model(pixel_values=pixel_values)
    last_hidden_state = vision_outputs.last_hidden_state
    print('last_hidden_state:', last_hidden_state.shape)

    # check InternCLIPModel
    from internvl_huggingface import InternCLIPModel
    from transformers import LlamaTokenizer

    model_path = '/mnt/petrelfs/share_data/wangwenhai/internvl/internvl_huggingface/intern_clip_13b'
    model = InternCLIPModel.from_pretrained(model_path).half().cuda()
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    tokenizer.add_eos_token = True
    texts = ['apple', 'cat', 'dog']
    tokenized = tokenizer(['summarize:' + item for item in texts], return_tensors='pt')
    input_ids = tokenized.input_ids.cuda()
    attention_mask = tokenized.attention_mask.cuda()
    pixel_values = torch.rand(1, 3, 224, 224).half().cuda()
    outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
    logits_per_image = outputs.logits_per_image
    logits_per_text = outputs.logits_per_text
    print('logits_per_image:', logits_per_image.shape)
    print('logits_per_text:', logits_per_text.shape)


if __name__ == '__main__':
    # save vit 6b model
    # save_intern_vision_model()

    # check preprocessor
    # check_preprocessor()

    # save intern clip model
    # save_intern_clip_model()

    # save tokenizer
    # save_tokenizer()

    # check_huggingface_model()
    check_readme()
