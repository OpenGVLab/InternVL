# How to Evaluate InternVL-Chat-V1-5 using VLMEvalKit?

In this tutorial, we will provide a detailed guide on how to evaluate InternVL-Chat-V1-5 using VLMEvalKit.

First of all, please follow this [guide](https://github.com/open-compass/VLMEvalKit/blob/main/Quickstart.md) to install VLMEvalKit.

## MMBench_DEV_EN

```
torchrun --nproc-per-node=8 run.py --data MMBench_DEV_EN --model InternVL-Chat-V1-5 --verbose
```

The result is:

```
2024-04-29 18:24:58,395 - Evaluation - INFO -   split   Overall  ...  spatial_relationship  structuralized_imagetext_understanding
0   dev  0.808419  ...              0.422222                                0.628205
```

## MMBench_DEV_CN

```
torchrun --nproc-per-node=8 run.py --data MMBench_DEV_CN --model InternVL-Chat-V1-5 --verbose
```

The result is:

```
2024-04-29 18:26:05,209 - Evaluation - INFO -   split   Overall  ...  spatial_relationship  structuralized_imagetext_understanding
0   dev  0.803265  ...              0.377778                                0.615385
```

## MMStar

```
torchrun --nproc-per-node=8 run.py --data MMStar --model InternVL-Chat-V1-5 --verbose
```

The result is:

```
2024-04-29 18:21:56,491 - Evaluation - INFO -   split   Overall  ...   math  science & technology
0  none  0.572667  ...  0.564                 0.408
```

## MME

```
torchrun --nproc-per-node=8 run.py --data MME --model InternVL-Chat-V1-5 --verbose
```

The result is:

```
2024-04-29 18:42:43,864 - Evaluation - INFO - Score:
2024-04-29 18:42:43,864 - Evaluation - INFO -     perception   reasoning    OCR  ...     posters  scene  text_translation
0  1641.915766  519.642857  147.5  ...  171.768707  156.5             185.0
```

## SEEDBench_IMG

```
torchrun --nproc-per-node=8 run.py --data SEEDBench_IMG --model InternVL-Chat-V1-5 --verbose
```

The result is:

```
2024-04-29 18:58:54,973 - Evaluation - INFO - Score:
2024-04-29 18:58:54,973 - Evaluation - INFO -   split   Overall  ...  Text Understanding  Visual Reasoning
0  none  0.757167  ...            0.440476          0.806647
```

## MMVet

```
torchrun --nproc-per-node=8 run.py --data MMVet --model InternVL-Chat-V1-5 --verbose
```

The result is:

```
2024-04-29 18:32:38,748 - Evaluation - INFO - Score:
2024-04-29 18:32:38,748 - Evaluation - INFO -   Category  tot        acc
0      rec  187  61.818182
1      ocr  108  68.981481
2     know   84  46.428571
3      gen   80  44.875000
4     spat   75  63.600000
5     math   26  62.307692
6  Overall  218  61.513761
```

Note that because the version of GPT used for scoring differs from the official server, the scores tested by VLMEvalKit will be slightly different.

## MMMU_DEV_VAL

```
torchrun --nproc-per-node=8 run.py --data MMMU_DEV_VAL --model InternVL-Chat-V1-5 --verbose
```

The result is:

```
2024-04-29 18:20:04,977 - Evaluation - INFO -         split  Overall  ...  Science  Tech & Engineering
0         dev     0.48  ...     0.36            0.428571
1  validation     0.45  ...     0.38            0.371429
```

## MathVista_MINI

```
torchrun --nproc-per-node=8 run.py --data MathVista_MINI --model InternVL-Chat-V1-5 --verbose
```

The result is:

```
2024-04-29 18:39:25,736 - Evaluation - INFO -                      Task&Skill   tot  prefetch  hit  prefetch_rate        acc
0                       Overall  1000       545  521      54.500000  52.100000
1          scientific reasoning   122        89   70      72.950820  57.377049
2   textbook question answering   158       101   86      63.924051  54.430380
3           numeric commonsense   144        39   41      27.083333  28.472222
4          arithmetic reasoning   353       147  198      41.643059  56.090652
5     visual question answering   179        91   88      50.837989  49.162011
6            geometry reasoning   239       144   94      60.251046  39.330544
7           algebraic reasoning   281       170  109      60.498221  38.790036
8      geometry problem solving   208       135   79      64.903846  37.980769
9             math word problem   186        70  118      37.634409  63.440860
10            logical reasoning    37        18    5      48.648649  13.513514
11    figure question answering   269       148  150      55.018587  55.762082
12        statistical reasoning   301       143  196      47.508306  65.116279
```

Note that because the version of GPT used for answer extraction differs from the official code, the scores tested by VLMEvalKit will be slightly different.

## ScienceQA_TEST

```
torchrun --nproc-per-node=8 run.py --data ScienceQA_TEST --model InternVL-Chat-V1-5 --verbose
```

The result is:

```
2024-04-29 19:10:03,279 - Evaluation - INFO - Score:
2024-04-29 19:10:03,279 - Evaluation - INFO -   split   Overall  ...  Weather and climate  World religions
0  test  0.940506  ...             0.948276              1.0
```

## HallusionBench

```
torchrun --nproc-per-node=8 run.py --data HallusionBench --model InternVL-Chat-V1-5 --verbose
```

The result is:

```
2024-04-29 18:21:37,606 - Evaluation - INFO - Score:
2024-04-29 18:21:37,606 - Evaluation - INFO -           split       aAcc       fAcc       qAcc
0       Overall  66.771819  40.173410  40.879121
1            VS  71.944444  40.517241  51.123596
2            VD  63.620981  40.000000  34.296029
3        VS_ocr  59.259259  38.461538  22.222222
4      VD_video  51.176471  10.416667  13.043478
5        VS_map  56.250000  18.181818  18.750000
6      VS_chart  81.538462  50.000000  72.368421
7      VS_table  75.892857  46.428571  55.813953
8     VD_figure  77.500000  65.853659  53.846154
9   VD_illusion  66.666667  41.935484  34.722222
10      VD_math  56.481481  25.000000  27.777778
11       VD_ocr  78.651685  58.139535  58.139535
```

## TextVQA_VAL

```
torchrun --nproc-per-node=8 run.py --data TextVQA_VAL --model InternVL-Chat-V1-5 --verbose
```

The result is:

```
2024-04-29 18:41:32,873 - Evaluation - INFO - VQA Eval Finished. Saved to ./InternVL-Chat-V1-5/InternVL-Chat-V1-5_TextVQA_VAL_acc.csv.
2024-04-29 18:41:32,873 - Evaluation - INFO -    Overall
0   80.488
```

## ChartQA_TEST

```
torchrun --nproc-per-node=8 run.py --data ChartQA_TEST --model InternVL-Chat-V1-5 --verbose
```

The result is:

```
2024-04-29 18:44:05,458 - Evaluation - INFO - VQA Eval Finished. Saved to ./InternVL-Chat-V1-5/InternVL-Chat-V1-5_ChartQA_TEST_acc.csv.
2024-04-29 18:44:05,458 - Evaluation - INFO -    test_human  test_augmented  Overall
0       73.04           94.32    83.68
```

## AI2D_TEST

```
torchrun --nproc-per-node=8 run.py --data AI2D_TEST --model InternVL-Chat-V1-5 --verbose
```

The result is:

```
2024-04-29 19:02:17,402 - Evaluation - INFO - Score:
2024-04-29 19:02:17,402 - Evaluation - INFO -   split   Overall  atomStructure  ...   typesOf  volcano  waterCNPCycle
0  none  0.806995           0.75  ...  0.752187      1.0       0.727273
```

## LLaVABench

```
torchrun --nproc-per-node=8 run.py --data LLaVABench --model InternVL-Chat-V1-5 --verbose
```

The result is:

```
Processing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:00 60/60 100%
     split  Relative Score (main)  VLM Score  GPT4 Score
0  overall                   82.0       63.7        77.7
1     conv                   82.9       74.1        89.4
2   detail                   72.0       48.0        66.7
3  complex                   86.0       65.7        76.4
```

## DocVQA_VAL

```
torchrun --nproc-per-node=8 run.py --data DocVQA_VAL --model InternVL-Chat-V1-5 --verbose
```

The result is:

```
2024-04-29 19:18:54,661 - Evaluation - INFO - VQA Eval Finished. Saved to ./InternVL-Chat-V1-5/InternVL-Chat-V1-5_DocVQA_VAL_acc.csv.
2024-04-29 19:18:54,661 - Evaluation - INFO -          val    Overall
0  90.500323  90.500323
```

## InfoVQA_VAL

```
torchrun --nproc-per-node=8 run.py --data InfoVQA_VAL --model InternVL-Chat-V1-5 --verbose
```

The result is:

```
2024-04-29 18:44:50,851 - Evaluation - INFO - VQA Eval Finished. Saved to ./InternVL-Chat-V1-5/InternVL-Chat-V1-5_InfoVQA_VAL_acc.csv.
2024-04-29 18:44:50,851 - Evaluation - INFO -          val    Overall
0  71.920408  71.920408
```

## OCRBench

```
torchrun --nproc-per-node=8 run.py --data OCRBench --model InternVL-Chat-V1-5 --verbose
```

The result is:

```
2024-04-29 18:56:05,437 - Evaluation - INFO - Score:
2024-04-29 18:56:05,437 - Evaluation - INFO - Text Recognition:238
2024-04-29 18:56:05,437 - Evaluation - INFO - Scene Text-centric VQA:178
2024-04-29 18:56:05,437 - Evaluation - INFO - Doc-oriented VQA:151
2024-04-29 18:56:05,438 - Evaluation - INFO - Key Information Extraction:153
2024-04-29 18:56:05,438 - Evaluation - INFO - Handwritten Mathematical Expression Recognition:4
2024-04-29 18:56:05,438 - Evaluation - INFO - Final Score:724
2024-04-29 18:56:05,438 - Evaluation - INFO - Final Score Norm:72.4
```

## RealWorldQA

```
torchrun --nproc-per-node=8 run.py --data RealWorldQA --model InternVL-Chat-V1-5 --verbose
```

The result is:

```
2024-04-29 18:56:43,192 - Evaluation - INFO - Score:
2024-04-29 18:56:43,192 - Evaluation - INFO -   split   Overall
0  none  0.660131
```

## SEEDBench2_Plus

```
torchrun --nproc-per-node=8 run.py --data SEEDBench2_Plus --model InternVL-Chat-V1-5 --verbose
```

The result is:

```
2024-05-29 12:41:47,313 - Evaluation - INFO -   split   Overall     chart       map      web
0  none  0.666227  0.650617  0.574969  0.79697
```
