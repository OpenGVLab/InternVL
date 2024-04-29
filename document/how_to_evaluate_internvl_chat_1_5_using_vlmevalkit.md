# How to Evaluate InternVL-Chat-V1-5 using VLMEvalKit?

In this tutorial, we will provide a detailed guide on how to evaluate InternVL-Chat-V1-5 using VLMEvalKit.

First of all, please follow this [guide](https://github.com/open-compass/VLMEvalKit/blob/main/Quickstart.md) to install VLMEvalKit.

## MMBench_DEV_EN

```
torchrun --nproc-per-node=8 run.py --data MMBench_DEV_EN --model InternVL-Chat-V1-5 --verbose
```

## MMBench_DEV_TEST

```
torchrun --nproc-per-node=8 run.py --data MMBench_DEV_TEST --model InternVL-Chat-V1-5 --verbose
```

## MMBench_DEV_CN

```
torchrun --nproc-per-node=8 run.py --data MMBench_DEV_CN --model InternVL-Chat-V1-5 --verbose
```

## MMBench_DEV_CN

```
torchrun --nproc-per-node=8 run.py --data MMBench_DEV_CN --model InternVL-Chat-V1-5 --verbose
```

## MMStar

```
torchrun --nproc-per-node=8 run.py --data MMStar --model InternVL-Chat-V1-5 --verbose
```

## MME

```
torchrun --nproc-per-node=8 run.py --data MME --model InternVL-Chat-V1-5 --verbose
```

## SEEDBench_IMG

```
torchrun --nproc-per-node=8 run.py --data SEEDBench_IMG --model InternVL-Chat-V1-5 --verbose
```

## MMVet

```
torchrun --nproc-per-node=8 run.py --data MMVet --model InternVL-Chat-V1-5 --verbose
```

## MMMU_DEV_VAL

```
torchrun --nproc-per-node=8 run.py --data MMMU_DEV_VAL --model InternVL-Chat-V1-5 --verbose
```

## MathVista_MINI

```
torchrun --nproc-per-node=8 run.py --data MathVista_MINI --model InternVL-Chat-V1-5 --verbose
```

## ScienceQA_TEST

```
torchrun --nproc-per-node=8 run.py --data ScienceQA_TEST --model InternVL-Chat-V1-5 --verbose
```

### HallusionBench

```
torchrun --nproc-per-node=8 run.py --data HallusionBench --model InternVL-Chat-V1-5 --verbose
```

### TextVQA_VAL

```
torchrun --nproc-per-node=8 run.py --data TextVQA_VAL --model InternVL-Chat-V1-5 --verbose
```

### ChartQA_TEST

```
torchrun --nproc-per-node=8 run.py --data ChartQA_TEST --model InternVL-Chat-V1-5 --verbose
```

### AI2D_TEST

```
torchrun --nproc-per-node=8 run.py --data AI2D_TEST --model InternVL-Chat-V1-5 --verbose
```

### LLaVABench

```
torchrun --nproc-per-node=8 run.py --data LLaVABench --model InternVL-Chat-V1-5 --verbose
```

### DocVQA_VAL

```
torchrun --nproc-per-node=8 run.py --data DocVQA_VAL --model InternVL-Chat-V1-5 --verbose
```

### InfoVQA_VAL

```
torchrun --nproc-per-node=8 run.py --data DocVQA_VAL --model InternVL-Chat-V1-5 --verbose
```

### OCRBench

```
torchrun --nproc-per-node=8 run.py --data DocVQA_VAL --model InternVL-Chat-V1-5 --verbose
```

### RealWorldQA

```
torchrun --nproc-per-node=8 run.py --data RealWorldQA --model InternVL-Chat-V1-5 --verbose
```
