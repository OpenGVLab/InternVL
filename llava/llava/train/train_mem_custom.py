# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

replace_llama_attn_with_flash_attn()

from llava.train.train_custom import train
from llava.train.dist_utils import init_dist

if __name__ == "__main__":
    try:
        init_dist(launcher='slurm', backend='nccl')
        print("slurm environment detected")
    except:
        init_dist(launcher='pytorch', backend='nccl')
    train()
