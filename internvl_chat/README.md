## InternVL

### 1. 环境

srun命令缩写

```shell
alias s1ahusky2='srun -p fccd1f66-ac3a-48bd-b382-085be1e60c7d --workspace-id bb502ff7-ca20-439c-bc55-6e59e38d476f -N 1 -r N3lS.Ii.I60.8 -f pytorch --container-image registry.cn-sh-01.sensecore.cn/foundation-ccr/base:20230719-03h55m43s  --container-mounts ef9e6157-1f8e-11ee-88d0-c6880f6d70d9:/mnt/afs'
alias s8ahusky2='srun -p fccd1f66-ac3a-48bd-b382-085be1e60c7d --workspace-id bb502ff7-ca20-439c-bc55-6e59e38d476f -N 1 -r N3lS.Ii.I60.8 -f pytorch --container-image registry.cn-sh-01.sensecore.cn/foundation-ccr/base:20230719-03h55m43s  --container-mounts ef9e6157-1f8e-11ee-88d0-c6880f6d70d9:/mnt/afs'
alias s16ahusky2='srun -p fccd1f66-ac3a-48bd-b382-085be1e60c7d --workspace-id bb502ff7-ca20-439c-bc55-6e59e38d476f -N 2 -r N3lS.Ii.I60.8 -f pytorch -d AllReduce --container-image registry.cn-sh-01.sensecore.cn/foundation-ccr/base:20230719-03h55m43s  --container-mounts ef9e6157-1f8e-11ee-88d0-c6880f6d70d9:/mnt/afs'
alias s32ahusky2='srun -p fccd1f66-ac3a-48bd-b382-085be1e60c7d --workspace-id bb502ff7-ca20-439c-bc55-6e59e38d476f -N 4 -r N3lS.Ii.I60.8 -f pytorch -d AllReduce --container-image registry.cn-sh-01.sensecore.cn/foundation-ccr/base:20230719-03h55m43s  --container-mounts ef9e6157-1f8e-11ee-88d0-c6880f6d70d9:/mnt/afs'
alias s64ahusky2='srun -p fccd1f66-ac3a-48bd-b382-085be1e60c7d --workspace-id bb502ff7-ca20-439c-bc55-6e59e38d476f -N 8 -r N3lS.Ii.I60.8 -f pytorch -d AllReduce --container-image registry.cn-sh-01.sensecore.cn/foundation-ccr/base:20230719-03h55m43s  --container-mounts ef9e6157-1f8e-11ee-88d0-c6880f6d70d9:/mnt/afs'
alias s128ahusky2='srun -p fccd1f66-ac3a-48bd-b382-085be1e60c7d --workspace-id bb502ff7-ca20-439c-bc55-6e59e38d476f -N 16 -r N3lS.Ii.I60.8 -f pytorch -d AllReduce --container-image registry.cn-sh-01.sensecore.cn/foundation-ccr/base:20230719-03h55m43s  --container-mounts ef9e6157-1f8e-11ee-88d0-c6880f6d70d9:/mnt/afs'
```

conda环境

```shell
conda activate /mnt/afs/user/chenzhe/.conda/envs/husky
```

### 2. 数据

Stage2 - QLLAMA数据：

/mnt/afs/user/chenzhe/workspace/InternVL/Husky2/data/internvl_data

Stage3 - Chat数据：

/mnt/afs/user/chenzhe/workspace/InternVL/Husky2/data/sft_data

现有数据：

https://ewt39kuao0.feishu.cn/wiki/N8K0wxIkniBpSJkQzrwcNJRonrf

### 3. 训练

32卡训练：

命令类似这样，**注意用你的绝对路径**

```shell
s32ahusky2 bash /mnt/afs/user/chenzhe/workspace/InternVL/Husky2/zh_shell/internvl_stage3_chat_v2_vicuna_train_qllama.sh
```

### 4. 测试MME

```shell
# 注意第一条命令的结尾不要有/
python eval.py --template "vicuna_v1.1" --model_path /mnt/afs/user/chenzhe/workspace/InternVL/Husky2/work_dirs/xxxxx
# 计算准确率，把刚生成的结果的路径传进去
python calculation.py --results_dir ./xxxxx
```

### 5. 要看的代码

https://github.com/czczup/InternVL/blob/master/Husky2/husky/train/internvl_stage3_chat_v2.py

https://github.com/czczup/InternVL/blob/master/Husky2/husky/model/internvl_hf_stage3_v2/modeling_intern_chat.py

https://github.com/czczup/InternVL/blob/master/Husky2/zh_shell/internvl_stage3_chat_v2_vicuna_train_qllama_exp1.sh

注意这两个路径要改成你们自己的

<img width="757" alt="image" src="https://github.com/czczup/InternVL/assets/23737120/eb59d57f-e00e-4bf6-96e1-22c93afce084">

### 6. 飞书

TODO List: https://ewt39kuao0.feishu.cn/wiki/UmDDwMCDDi75cSkLkXycizVgnqh

实验记录: https://ewt39kuao0.feishu.cn/wiki/D726wnWm3iAwqtkNVy5cpzFQn7b
