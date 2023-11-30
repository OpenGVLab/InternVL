from typing import Dict

import torch


class DictTensor:
    """
    enable to do `tokenizer(texts).to(device)`
    """

    def __init__(self, d: Dict[str, torch.Tensor]):
        self.d = d

    def to(self, device):
        return {k: v.to(device) for k, v in self.d.items()}


class JaCLIPForBenchmark:
    """
    enable to do model.encode_text(dict_tensor)
    """

    def __init__(self, model):
        self.model = model

    def encode_text(self, dict_tensor):
        return self.model.get_text_features(**dict_tensor)

    def encode_image(self, image):
        return self.model.get_image_features(image)


def load_japanese_clip(pretrained: str, device='cpu', **kwargs):
    """
    Load Japanese CLIP/CLOOB by rinna (https://github.com/rinnakk/japanese-clip)
    Remarks:
     - You must input not only input_ids but also attention_masks and position_ids when doing `model.encode_text()` to make it work correctly.
    """
    try:
        import japanese_clip as ja_clip
    except ImportError:
        raise ImportError('Install `japanese_clip` by `pip install git+https://github.com/rinnakk/japanese-clip.git`')
    cache_dir = kwargs.pop('cache_dir', None)
    model, transform = ja_clip.load(pretrained, device=device, cache_dir=cache_dir)

    class JaTokenizerForBenchmark:
        def __init__(self, ):
            self.tokenizer = ja_clip.load_tokenizer()

        def __call__(self, texts) -> Dict[str, torch.Tensor]:
            inputs = ja_clip.tokenize(texts, tokenizer=self.tokenizer, device='cpu')
            return DictTensor(inputs)

        def __len__(self):
            return len(self.tokenizer)

    return JaCLIPForBenchmark(model), transform, JaTokenizerForBenchmark()
