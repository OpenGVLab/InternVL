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

- Install dependencies using `requirements.txt`:

  ```bash
  pip install -r requirements.txt
  ```

  By default, our `requirements.txt` file includes the following dependencies:

  - `-r requirements/internvl_chat.txt`
  - `-r requirements/streamlit_demo.txt`
  - `-r requirements/classification.txt`
  - `-r requirements/segmentation.txt`

  The `clip_benchmark.txt` is **not** included in the default installation. If you require the `clip_benchmark` functionality, please install it manually by running the following command:

  ```bash
  pip install -r requirements/clip_benchmark.txt
  ```

### Additional Instructions

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

- Install `mmcv-full==1.6.2` (optional, for `segmentation`):

  ```bash
  pip install -U openmim
  mim install mmcv-full==1.6.2
  ```

- Install `apex` (optional, for `segmentation`):

  ```bash
  git clone https://github.com/NVIDIA/apex.git
  git checkout 2386a912164b0c5cfcd8be7a2b890fbac5607c82  # https://github.com/NVIDIA/apex/issues/1735
  pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
  ```

  If you encounter `ModuleNotFoundError: No module named 'fused_layer_norm_cuda'`, it is because apex's CUDA extensions are not being installed successfully. You can try uninstalling apex and the code will default to the PyTorch version of RMSNorm. Alternatively, if you prefer using apex, try adding a few lines to `setup.py` and then recompiling.

  <img src=https://github.com/OpenGVLab/InternVL/assets/23737120/c04a989c-8024-49fa-b62c-2da623e63729 width=50%>
