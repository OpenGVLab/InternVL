## `internvl_custom2hf.py`

Convert model weights from a **custom format** to the **HuggingFace format**, with L2 distance comparison.

### Usage

```bash
python internvl_custom2hf.py \
  --custom_path /path/to/custom_ckpt \
  --hf_path /path/to/hf_model \
  --save_path /path/to/output_dir
```

### Arguments

* `--custom_path`: Directory containing `.safetensors` files from your custom-trained model.
* `--hf_path`: Path to a reference HuggingFace model (for config and tokenizer).
* `--save_path`: Output directory to save the converted HuggingFace-format model.

### Example

```bash
python internvl_custom2hf.py \
  --custom_path /path/to/your/InternVL3-8B-finetuned \
  --hf_path OpenGVLab/InternVL3-8B-hf \
  --save_path /path/to/your/InternVL3-8B-finetuned-hf
```

## `internvl_hf2custom.py`

Convert model weights from the **HuggingFace format** back to a **custom format**, with L2 distance comparison.

### Usage

```bash
python internvl_hf2custom.py \
  --custom_path /path/to/custom_model \
  --hf_path /path/to/hf_weights \
  --save_path /path/to/output_dir
```

### Arguments

* `--custom_path`: Path to the custom model config and tokenizer (used to rebuild the model and for comparison).
* `--hf_path`: Path to HuggingFace-format `.safetensors` files to be converted.
* `--save_path`: Output directory to save the converted custom-format model.

### Example

```bash
python internvl_hf2custom.py \
  --custom_path OpenGVLab/InternVL3-8B \
  --hf_path OpenGVLab/InternVL3-8B-hf \
  --save_path /path/to/your/InternVL3-8B-hf-to-custom
```
