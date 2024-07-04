# How to deploy a local demo?

## Launch a Controller

```shell
# run the command in the `internvl_chat_llava` folder
python -m llava.serve.controller --host 0.0.0.0 --port 10000
```

## Launch a Gradio Web Server

```shell
# run the command in the `internvl_chat_llava` folder
python -m llava.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload
```

## Launch a Model Worker

### Options

- `--host <host_address>`: Specifies the host address on which the model worker will run. Use "0.0.0.0" to allow connections from any IP address.
- `--controller <controller_address>`: Specifies the address of the controller node responsible for managing model deployment and execution.
- `--port <port_number>`: Specifies the port number on which the model worker will listen for incoming requests.
- `--worker <worker_address>`: Specifies the address of the worker node where the model will be executed.
- `--model-path <model_file_path>`: Specifies the file path to the machine learning model to be deployed and executed.

### Additional Options

#### Multi-GPU Deployment

To enable deployment on multiple GPUs, use the `--device auto` option. This allows the script to utilize all available GPU devices for model execution automatically.

#### Quantization Deployment

To enable quantization for model deployment, use the `--load-8bit` option. This performs quantization on the model, reducing its precision to 8 bits for improved efficiency.

__Note: The `--device auto` and `--load-8bit` options cannot be used simultaneously.__

```shell
# OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-7B
# run the command in the `internvl_chat_llava` folder
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-7B

# OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-13B
# run the command in the `internvl_chat_llava` folder
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40001 --worker http://localhost:40001 --model-path OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-13B

# OpenGVLab/InternVL-Chat-V1-1
# run the command in the `internvl_chat` folder
python -m internvl.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40002 --worker http://localhost:40002 --model-path OpenGVLab/InternVL-Chat-V1-1

# OpenGVLab/InternVL-Chat-V1-2
# run the command in the `internvl_chat` folder
python -m internvl.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40003 --worker http://localhost:40003 --model-path OpenGVLab/InternVL-Chat-V1-2

# OpenGVLab/InternVL-Chat-V1-2-Plus
# run the command in the `internvl_chat` folder
python -m internvl.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40004 --worker http://localhost:40004 --model-path OpenGVLab/InternVL-Chat-V1-2-Plus

# OpenGVLab/InternVL-Chat-V1-5
# run the command in the `internvl_chat` folder
python -m internvl.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40005 --worker http://localhost:40005 --model-path OpenGVLab/InternVL-Chat-V1-5
```
