from contextlib import suppress

import torch
import torch.nn.functional as F
from tqdm import tqdm


def evaluate(model, dataloader, tokenizer, device, amp=True, recall_k_list=[5]):
    """
    Evaluate the model on the given dataset

    Parameters
    ----------

    model: torch.nn,Module
        CLIP-like model with `encode_image` and `encode_text`

    dataloader: torch.utils.data.Dataloader
        dataloader to use for evaluation

    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers

    device: cpu/cuda

    amp: whether to use automatic mixed precision

    recall_k_list: list of int
        recall@k k's to use

    Returns
    -------

    dict of retrieval metrics
    """
    # list of batch of images embedding
    batch_images_emb_list = []
    batch_images_feat_list = []
    # list of batch of text embedding
    batch_texts_emb_list = []
    batch_texts_list = []
    # for each text, we collect the corresponding image index, as each image can have multiple corresponding texts
    texts_image_index = []
    dataloader = dataloader_with_indices(dataloader)
    autocast = torch.cuda.amp.autocast if amp else suppress
    for batch_images, batch_texts, inds in tqdm(dataloader):
        batch_images = batch_images.to(device)
        # tokenize all texts in the batch
        batch_texts_tok = tokenizer([text for i, texts in enumerate(batch_texts) for text in texts]).to(device)
        temp = tokenizer([text for i, texts in enumerate(batch_texts) for text in texts], prefix='').to(device)
        batch_texts_list.append(temp)
        # store the index of image for each text
        batch_texts_image_index = [ind for ind, texts in zip(inds, batch_texts) for text in texts]

        # compute the embedding of images and texts
        with torch.no_grad(), autocast():
            batch_images_backbone_feat, batch_images_qformer_feat = model.get_image_features(
                pixel_values=batch_images, output_hidden_states=False, return_dict=True)
            batch_images_backbone_emb = model.clip_projector(batch_images_backbone_feat)
            batch_images_backbone_emb = F.normalize(batch_images_backbone_emb, dim=-1)

            batch_images_qformer_emb = model.clip_projector2(batch_images_qformer_feat)
            batch_images_qformer_emb = F.normalize(batch_images_qformer_emb, dim=-1)

            batch_images_emb = F.normalize(batch_images_backbone_emb + batch_images_qformer_emb, dim=-1)
            batch_texts_emb = F.normalize(model.encode_text(batch_texts_tok), dim=-1)

        batch_images_emb_list.append(batch_images_emb.cpu())
        batch_images_feat_list.append(batch_images_backbone_feat.cpu())
        batch_texts_emb_list.append(batch_texts_emb.cpu())
        texts_image_index.extend(batch_texts_image_index)

    batch_size = len(batch_images_emb_list[0])

    # concatenate all embeddings
    images_emb = torch.cat(batch_images_emb_list)
    images_feat = torch.cat(batch_images_feat_list).to(model.device)
    texts_emb = torch.cat(batch_texts_emb_list)
    texts_ids = torch.cat(batch_texts_list)

    # get the score for each text and image pair
    scores = texts_emb @ images_emb.t()  # (5000, 1000)
    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), texts_image_index] = True

    # I2T
    k_test = 32
    new_scores = -100 * torch.ones(scores.shape, device=scores.device)
    for i in tqdm(range(scores.size(1))):
        scores_i2t = scores[:, i]
        topk_sim, topk_idx = scores_i2t.topk(k=k_test, dim=0)
        image_inputs = images_feat[i].repeat(k_test, 1, 1)
        with torch.no_grad():
            score = model.encode_image_text(
                image_embeds=image_inputs,
                input_ids=texts_ids[topk_idx],
            ).to(scores.dtype)
        new_scores[topk_idx, i] = score.cpu() + scores[topk_idx, i]
    metrics = {}
    for recall_k in recall_k_list:
        metrics[f'text_retrieval_recall@{recall_k}'] = (
                batchify(recall_at_k, new_scores.T, positive_pairs.T, batch_size, device,
                         k=recall_k) > 0).float().mean().item()

    # T2I
    k_test = 32
    new_scores = -100 * torch.ones(scores.shape, device=scores.device)
    for i in tqdm(range(scores.size(0))):
        scores_t2i = scores[i, :]
        topk_sim, topk_idx = scores_t2i.topk(k=k_test, dim=0)
        image_inputs = images_feat[topk_idx]
        with torch.no_grad():
            score = model.encode_image_text(
                image_embeds=image_inputs,
                input_ids=texts_ids[i][None, :].repeat(k_test, 1),
            ).to(scores.dtype)
        new_scores[i, topk_idx] = score.cpu() + scores[i, topk_idx]

    for recall_k in recall_k_list:
        metrics[f'image_retrieval_recall@{recall_k}'] = (
                batchify(recall_at_k, new_scores, positive_pairs, batch_size, device,
                         k=recall_k) > 0).float().mean().item()

    return metrics


def dataloader_with_indices(dataloader):
    start = 0
    for x, y in dataloader:
        end = start + len(x)
        inds = torch.arange(start, end)
        yield x, y, inds
        start = end


def recall_at_k(scores, positive_pairs, k):
    """
    Compute the recall at k for each sample
    :param scores: compability score between  text and image embeddings (nb texts, nb images)
    :param k: number of images to consider per text, for retrieval
    :param positive_pairs: boolean matrix of positive pairs (nb texts, nb images)
    :return: recall at k averaged over all texts
    """
    nb_texts, nb_images = scores.shape
    # for each text, sort according to image scores in decreasing order
    topk_indices = torch.topk(scores, k, dim=1)[1]
    # compute number of positives for each text
    nb_positive = positive_pairs.sum(dim=1)
    # nb_texts, k, nb_images
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    # compute number of true positives
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    # a true positive means a positive among the topk
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1, 2))
    # compute recall at k
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k


def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)
