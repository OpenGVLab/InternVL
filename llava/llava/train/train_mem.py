# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
from llava.train.llama_tokenizer_monkey_patch import fix_llama_tokenizer_bug

replace_llama_attn_with_flash_attn()
fix_llama_tokenizer_bug()  # compatible with transformers==4.32.0

from llava.train.train import train

if __name__ == "__main__":
    train()
