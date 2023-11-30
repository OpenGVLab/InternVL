import open_clip


def load_open_clip(model_name: str = 'ViT-B-32-quickgelu', pretrained: str = 'laion400m_e32', cache_dir: str = None,
                   device='cpu'):
    model, _, transform = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, cache_dir=cache_dir)
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, transform, tokenizer
