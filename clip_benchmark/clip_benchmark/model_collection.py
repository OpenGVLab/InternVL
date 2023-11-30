import open_clip


def get_model_collection_from_file(path):
    return [l.strip().split(',') for l in open(path).readlines()]


model_collection = {
    'openclip_base': [
        ('ViT-B-32-quickgelu', 'laion400m_e32'),
        ('ViT-B-32', 'laion2b_e16'),
        ('ViT-B-32', 'laion2b_s34b_b79k'),
        ('ViT-B-16', 'laion400m_e32'),
        ('ViT-B-16-plus-240', 'laion400m_e32'),
        ('ViT-L-14', 'laion400m_e32'),
        ('ViT-L-14', 'laion2b_s32b_b82k'),
        ('ViT-H-14', 'laion2b_s32b_b79k'),
        ('ViT-g-14', 'laion2b_s12b_b42k'),
    ],
    'openclip_multilingual': [
        ('xlm-roberta-base-ViT-B-32', 'laion5b_s13b_b90k'),
        ('xlm-roberta-large-ViT-H-14', 'frozen_laion5b_s13b_b90k'),
    ],
    'openclip_all': open_clip.list_pretrained(),
    'openai': [
        ('ViT-B-32', 'openai'),
        ('ViT-B-16', 'openai'),
        ('ViT-L-14', 'openai'),
        ('ViT-L-14-336', 'openai'),
    ]
}
