## 🛠️ Installation

- Clone this repository:

  ```bash
  git clone https://github.com/OpenGVLab/InternVL.git
  ```

- Create a conda virtual environment and activate it:

  ```bash
  conda create -n internvl python=3.9 -y
  conda activate internvl
  ```

- Install `PyTorch>=2.0` and `torchvision>=0.15.2` with `CUDA>=11.6`:

  For examples, to install `torch==2.0.1` with `CUDA==11.8`:

  ```bash
  conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
  # or
  pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
  ```

- Install `flash-attn==2.3.6`:

  ```bash
  pip install flash-attn==2.3.6 --no-build-isolation
  ```

  Alternatively you can compile from source:

  ```bash
  git clone https://github.com/Dao-AILab/flash-attention.git
  cd flash-attention
  git checkout v2.3.6
  python setup.py install
  ```

- Install `timm==0.9.12` and `mmcv-full==1.6.2`:

  ```bash
  pip install timm==0.9.12
  pip install -U openmim
  mim install mmcv-full==1.6.2  # (optional, for mmsegmentation)
  ```

- Install `transformers==4.37.2`:

  ```bash
  pip install transformers==4.37.2
  ```

- Install `apex` (optional):

  ```bash
  git clone https://github.com/NVIDIA/apex.git
  git checkout 2386a912164b0c5cfcd8be7a2b890fbac5607c82  # https://github.com/NVIDIA/apex/issues/1735
  pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
  ```

  If you meet `ModuleNotFoundError: No module named 'fused_layer_norm_cuda'`, please note that this is because apex's CUDA extensions are not being installed successfully. You can try to uninstall apex and the code will default to the PyTorch version of RMSNorm; Or, if you want to use apex, try adding a few lines to `setup.py`, like this, and then recompiling.

  <img src=https://github.com/OpenGVLab/InternVL/assets/23737120/c04a989c-8024-49fa-b62c-2da623e63729 width=50%>

- Install other requirements:

  ```bash
  pip install opencv-python termcolor yacs pyyaml scipy
  pip install deepspeed==0.13.5
  pip install pycocoevalcap tqdm
  ```
