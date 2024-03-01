import json
import random
import re
from typing import Dict

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode


def build_transform(input_size):
    # match fine-tune setting with blip2
    # https://github.com/salesforce/LAVIS/blob/main/lavis/processors/blip_processors.py
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.RandomResizedCrop(input_size, scale=(0.5, 1.0),
                            interpolation=InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    return transform


class FlickrDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, metas, tokenizer, data_args):
        super(FlickrDataset, self).__init__()

        f = open(metas['annotation'])
        lines = f.readlines()[1:]

        self.data_args = data_args
        self.tokenizer = tokenizer
        self.images = []
        self.image_ids = []
        self.captions = []

        for line in lines:
            image, caption = line.strip().split('.jpg,')
            image_id = int(image)
            caption = self.process_single_caption(caption)
            image = image + '.jpg'
            image_path = metas['root'] + '/' + image
            self.images.append(image_path)
            self.image_ids.append(image_id)
            self.captions.append(caption)
        print(f'There are {len(self.images)} images.')
        print(f'There are {len(self.captions)} captions.')

    def __len__(self):
        return len(self.images)

    def process_single_caption(self, caption, max_words=50):
        caption = re.sub(r"([.!\"()*#:;~])", ' ', caption.lower())
        caption = re.sub(r'\s{2,}', ' ', caption)
        caption = caption.rstrip('\n')
        caption = caption.strip(' ')

        # truncate caption
        caption_words = caption.split(' ')
        if len(caption_words) > max_words:
            caption = ' '.join(caption_words[: max_words])
        return caption

    def preprocess(self, image, caption, neg_caption):
        model_inputs = dict()

        # input image
        image_transform = build_transform(input_size=self.data_args.force_image_size)
        image = Image.open(image)
        image = image.convert('RGB')
        pixel_values = image_transform(image)
        model_inputs['pixel_values'] = pixel_values

        # for image-text matching
        pos_model_inputs = self.tokenizer(
            caption,
            max_length=self.data_args.max_seq_length,
            padding='max_length' if self.data_args.pad_to_max_length else False,
            truncation=True,
            return_tensors='pt',
        )
        model_inputs['positive_input_ids'] = pos_model_inputs['input_ids']
        model_inputs['positive_attention_mask'] = pos_model_inputs['attention_mask']
        neg_model_inputs = self.tokenizer(
            neg_caption,
            max_length=self.data_args.max_seq_length,
            padding='max_length' if self.data_args.pad_to_max_length else False,
            truncation=True,
            return_tensors='pt',
        )
        model_inputs['negative_input_ids'] = neg_model_inputs['input_ids']
        model_inputs['negative_attention_mask'] = neg_model_inputs['attention_mask']

        # for image-text contrastive learning
        summarize_model_inputs = self.tokenizer(
            'summarize:' + caption,
            max_length=self.data_args.max_seq_length,
            padding='max_length' if self.data_args.pad_to_max_length else False,
            truncation=True,
            return_tensors='pt',
        )
        model_inputs['summarize_input_ids'] = summarize_model_inputs['input_ids']
        model_inputs['summarize_attention_mask'] = summarize_model_inputs['attention_mask']

        # for image-grounded text generation
        prefix = f'English caption:'
        content = caption
        tokenized_prefix = self.tokenizer(
            prefix, padding=False, truncation=True, return_tensors='pt',
        )
        prefix_input_ids = tokenized_prefix['input_ids'][:, :-1]  # remove eos
        prefix_attention_mask = tokenized_prefix['attention_mask'][:, :-1]  # remove eos
        tokenized_content = self.tokenizer(
            content,
            max_length=self.data_args.max_seq_length - prefix_input_ids.size(1) + 1,
            padding='max_length' if self.data_args.pad_to_max_length else False,
            truncation=True,
            return_tensors='pt',
        )
        content_input_ids = tokenized_content['input_ids'][:, 1:]  # remove bos
        content_attention_mask = tokenized_content['attention_mask'][:, 1:]  # remove bos
        model_inputs['input_ids'] = torch.cat([prefix_input_ids, content_input_ids], dim=1)
        model_inputs['attention_mask'] = torch.cat([prefix_attention_mask, content_attention_mask], dim=1)
        labels = model_inputs['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[:, :prefix_input_ids.size(1) - 1] = -100
        model_inputs['labels'] = labels
        return model_inputs

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        i = i % len(self.images)
        j = random.randint(0, len(self.images) - 1)
        while self.image_ids[j] == self.image_ids[i]:
            j = random.randint(0, len(self.images) - 1)
        ret = self.preprocess(self.images[i], self.captions[i], self.captions[j])
        # for image-text matching
        ret['positive_input_ids'] = ret['positive_input_ids'][0]
        ret['positive_attention_mask'] = ret['positive_attention_mask'][0]
        ret['negative_input_ids'] = ret['negative_input_ids'][0]
        ret['negative_attention_mask'] = ret['negative_attention_mask'][0]
        # for image-text contrastive learning
        ret['summarize_input_ids'] = ret['summarize_input_ids'][0]
        ret['summarize_attention_mask'] = ret['summarize_attention_mask'][0]
        # for image-grounded text generation
        ret['input_ids'] = ret['input_ids'][0]
        ret['attention_mask'] = ret['attention_mask'][0]
        ret['labels'] = ret['labels'][0]
        ret['image_ids'] = torch.Tensor([self.image_ids[i]]).long()
        return ret


class COCODataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, metas, tokenizer, data_args):
        super(COCODataset, self).__init__()

        annotations = json.load(open(metas['annotation']))

        self.data_args = data_args
        self.tokenizer = tokenizer
        self.images = []
        self.image_ids = []
        self.captions = []

        for annotation in annotations:
            image_id = int(annotation['image_id'].split('_')[-1])
            caption = annotation['caption']
            caption = self.process_single_caption(caption)
            image = annotation['image']
            image_path = metas['root'] + '/' + image
            self.images.append(image_path)
            self.image_ids.append(image_id)
            self.captions.append(caption)
        print(f'There are {len(self.images)} images.')
        print(f'There are {len(self.captions)} captions.')

    def __len__(self):
        return len(self.images)

    def process_single_caption(self, caption, max_words=50):
        caption = re.sub(r"([.!\"()*#:;~])", ' ', caption.lower())
        caption = re.sub(r'\s{2,}', ' ', caption)
        caption = caption.rstrip('\n')
        caption = caption.strip(' ')

        # truncate caption
        caption_words = caption.split(' ')
        if len(caption_words) > max_words:
            caption = ' '.join(caption_words[: max_words])
        return caption

    def preprocess(self, image, caption, neg_caption):
        model_inputs = dict()

        # input image
        image_transform = build_transform(input_size=self.data_args.force_image_size)
        image = Image.open(image)
        image = image.convert('RGB')
        pixel_values = image_transform(image)
        model_inputs['pixel_values'] = pixel_values

        # for image-text matching
        pos_model_inputs = self.tokenizer(
            caption,
            max_length=self.data_args.max_seq_length,
            padding='max_length' if self.data_args.pad_to_max_length else False,
            truncation=True,
            return_tensors='pt',
        )
        model_inputs['positive_input_ids'] = pos_model_inputs['input_ids']
        model_inputs['positive_attention_mask'] = pos_model_inputs['attention_mask']
        neg_model_inputs = self.tokenizer(
            neg_caption,
            max_length=self.data_args.max_seq_length,
            padding='max_length' if self.data_args.pad_to_max_length else False,
            truncation=True,
            return_tensors='pt',
        )
        model_inputs['negative_input_ids'] = neg_model_inputs['input_ids']
        model_inputs['negative_attention_mask'] = neg_model_inputs['attention_mask']

        # for image-text contrastive learning
        summarize_model_inputs = self.tokenizer(
            'summarize:' + caption,
            max_length=self.data_args.max_seq_length,
            padding='max_length' if self.data_args.pad_to_max_length else False,
            truncation=True,
            return_tensors='pt',
        )
        model_inputs['summarize_input_ids'] = summarize_model_inputs['input_ids']
        model_inputs['summarize_attention_mask'] = summarize_model_inputs['attention_mask']

        # for image-grounded text generation
        prefix = f'English caption:'
        content = caption
        tokenized_prefix = self.tokenizer(
            prefix, padding=False, truncation=True, return_tensors='pt',
        )
        prefix_input_ids = tokenized_prefix['input_ids'][:, :-1]  # remove eos
        prefix_attention_mask = tokenized_prefix['attention_mask'][:, :-1]  # remove eos
        tokenized_content = self.tokenizer(
            content,
            max_length=self.data_args.max_seq_length - prefix_input_ids.size(1) + 1,
            padding='max_length' if self.data_args.pad_to_max_length else False,
            truncation=True,
            return_tensors='pt',
        )
        content_input_ids = tokenized_content['input_ids'][:, 1:]  # remove bos
        content_attention_mask = tokenized_content['attention_mask'][:, 1:]  # remove bos
        model_inputs['input_ids'] = torch.cat([prefix_input_ids, content_input_ids], dim=1)
        model_inputs['attention_mask'] = torch.cat([prefix_attention_mask, content_attention_mask], dim=1)
        labels = model_inputs['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[:, :prefix_input_ids.size(1) - 1] = -100
        model_inputs['labels'] = labels
        return model_inputs

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        i = i % len(self.images)
        j = random.randint(0, len(self.images) - 1)
        while self.image_ids[j] == self.image_ids[i]:
            j = random.randint(0, len(self.images) - 1)
        ret = self.preprocess(self.images[i], self.captions[i], self.captions[j])
        # for image-text matching
        ret['positive_input_ids'] = ret['positive_input_ids'][0]
        ret['positive_attention_mask'] = ret['positive_attention_mask'][0]
        ret['negative_input_ids'] = ret['negative_input_ids'][0]
        ret['negative_attention_mask'] = ret['negative_attention_mask'][0]
        # for image-text contrastive learning
        ret['summarize_input_ids'] = ret['summarize_input_ids'][0]
        ret['summarize_attention_mask'] = ret['summarize_attention_mask'][0]
        # for image-grounded text generation
        ret['input_ids'] = ret['input_ids'][0]
        ret['attention_mask'] = ret['attention_mask'][0]
        ret['labels'] = ret['labels'][0]
        ret['image_ids'] = torch.Tensor([self.image_ids[i]]).long()
        return ret
