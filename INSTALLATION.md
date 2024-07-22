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

### Additional Instructions for Optional Dependencies

- Install `apex` (optional, if you want to use `FusedRMSNorm`):

  ```bash
  git clone https://github.com/NVIDIA/apex.git
  git checkout 2386a912164b0c5cfcd8be7a2b890fbac5607c82  # https://github.com/NVIDIA/apex/issues/1735
  pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
  ```

  If you encounter `ModuleNotFoundError: No module named 'fused_layer_norm_cuda'`, it is because apex's CUDA extensions are not being installed successfully. You can try uninstalling apex and the code will default to the PyTorch version of RMSNorm. Alternatively, if you prefer using apex, try adding a few lines to `setup.py` and then recompiling.

  <img src=https://github.com/OpenGVLab/InternVL/assets/23737120/c04a989c-8024-49fa-b62c-2da623e63729 width=50%>
