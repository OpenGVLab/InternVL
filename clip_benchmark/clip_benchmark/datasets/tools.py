import re


def process_single_caption(caption, max_words=50):
    caption = re.sub(r"([.!\"()*#:;~])", ' ', caption.lower())
    caption = re.sub(r'\s{2,}', ' ', caption)
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[: max_words])
    return caption


def pre_caption(caption, max_words=50):
    if type(caption) == str:
        caption = process_single_caption(caption, max_words)
    else:
        caption = [process_single_caption(c, max_words) for c in caption]
    return caption
