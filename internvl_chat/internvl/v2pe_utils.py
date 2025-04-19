import torch
import torch.nn as nn
import random
# get rope pos id while evaluating
def get_rope_pos_id(ret, dtype, rope_pos_id_version='default', position_id=None,
                    IMG_START_TOKEN='<img>',IMG_END_TOKEN='</img>',rope_pos_id_stride=None, tokenizer=None, num_image_token=256):
    image_start_token_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
    image_end_token_id = tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
    rope_pos_id_list = []
    assert ret['input_ids'].shape[0] == 1, 'batch size should be 1, other batch sizes are not supported yet'
    input_ids_0 = ret['input_ids'][0]
    attention_mask_0 = ret['attention_mask'][0]
    image_start_token_id_idxs = torch.where(input_ids_0 == image_start_token_id)[0]
    image_end_token_id_idxs = torch.where(input_ids_0 == image_end_token_id)[0]
    num_tiles = (image_end_token_id_idxs - image_start_token_id_idxs) // num_image_token
    last_record_pos_id = -1
    start_index = 0

    assert rope_pos_id_version in ['v2pe_fix', 'v2pe_rnd', 'default'], f'{rope_pos_id_version} not supported for eval'

    for i in range(len(image_start_token_id_idxs)):

        num_tile = num_tiles[i]

        rope_pos_id_pre = attention_mask_0[start_index:image_start_token_id_idxs[i] + 1].long().cumsum(-1) - 1 + (last_record_pos_id + 1)
        rope_pos_id_pre.masked_fill_(attention_mask_0[start_index:image_start_token_id_idxs[i] + 1] == 0, 1)
        rope_pos_id_list.append(rope_pos_id_pre)

        last_record_pos_id = rope_pos_id_pre[-1].long()

        if rope_pos_id_version == 'v2pe_fix':
            assert rope_pos_id_stride is not None, 'when rope_pos_id_version is fix, self.rope_pos_id_stride should not be None'
            small_stride = rope_pos_id_stride / num_image_token
            split_img_id_idxs = torch.linspace(last_record_pos_id,last_record_pos_id+small_stride*(num_image_token * num_tile ),(num_image_token * num_tile + 1))[1:].to(dtype=dtype)
            rope_pos_id_list.append(split_img_id_idxs)
            last_record_pos_id = torch.ceil(split_img_id_idxs[-1]).long()
        elif rope_pos_id_version == 'v2pe_rnd':
            random_from=[1,2,4,8,16,32,64,128,256]
            rope_pos_id_stride=random.choice(random_from)
            small_stride = rope_pos_id_stride / num_image_token
            split_img_id_idxs = torch.linspace(last_record_pos_id,last_record_pos_id+small_stride*(num_image_token * num_tile ),(num_image_token * num_tile + 1))[1:].to(dtype=dtype)
            rope_pos_id_list.append(split_img_id_idxs)
            last_record_pos_id = torch.ceil(split_img_id_idxs[-1]).long()
        elif rope_pos_id_version == 'default':
            split_img_id_idxs = torch.linspace(last_record_pos_id,
                                               last_record_pos_id + (num_tile - 1) * num_image_token,
                                               (num_tile - 1) * num_image_token + 1)[1:].to(dtype=dtype)
            rope_pos_id_list.append(split_img_id_idxs)
            thumbnail_id_idxs = torch.linspace(last_record_pos_id + (num_tile - 1) * num_image_token,
                                               last_record_pos_id + num_tile * num_image_token,
                                               num_image_token + 1)[1:].to(dtype=dtype)
            rope_pos_id_list.append(thumbnail_id_idxs)
            last_record_pos_id = (last_record_pos_id + num_tile * num_image_token).long()
        else:
            raise NotImplementedError(f'not implement for {rope_pos_id_version}')

        start_index = image_start_token_id_idxs[i] + num_tile * num_image_token + 1
        assert input_ids_0[start_index] == image_end_token_id
        assert start_index == image_end_token_id_idxs[i]

    assert image_end_token_id_idxs[-1] == start_index
    rope_pos_id_pre = attention_mask_0[start_index:].long().cumsum(-1) - 1 + (last_record_pos_id + 1)
    rope_pos_id_pre.masked_fill_(attention_mask_0[start_index:] == 0, 1)
    rope_pos_id_list.append(rope_pos_id_pre)

    rope_pos_id_list=[_.to('cpu') for _ in rope_pos_id_list]
    rope_pos_id = torch.cat(rope_pos_id_list).to(dtype=dtype)
    if rope_pos_id_version == 'default':
        rope_pos_id = rope_pos_id.long()
        assert torch.equal(rope_pos_id, position_id.to(rope_pos_id.device)), (rope_pos_id, position_id.to(rope_pos_id.device))
        assert torch.allclose(rope_pos_id, position_id.to(rope_pos_id.device), atol=1e-32)

    assert rope_pos_id.shape == input_ids_0.shape

    return list(rope_pos_id.numpy())


# Rotary Position Embedding for V2PE
class V2PE(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.inv_freq = None
        # inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        # self.register_buffer('inv_freq', inv_freq, persistent=False)

        self.max_seq_len_cached = -1 

    def _set_cos_sin_cache(self, pos_id, device, dtype):
        if self.inv_freq is None:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim))
            del self.inv_freq
            self.register_buffer('inv_freq', inv_freq, persistent=False)

        pos_id=pos_id.squeeze(0)
        freqs = torch.outer(pos_id, self.inv_freq.to(device=pos_id.device))

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos().to(dtype), persistent=False)
        self.register_buffer('sin_cached', emb.sin().to(dtype), persistent=False)

    def forward(self, x, global_posid=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        self._set_cos_sin_cache(pos_id=global_posid, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:].to(dtype=x.dtype),
            self.sin_cached[:].to(dtype=x.dtype),
        )