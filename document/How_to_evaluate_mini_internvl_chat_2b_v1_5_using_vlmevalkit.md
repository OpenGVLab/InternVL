# How to Evaluate Mini-InternVL-Chat-2B-V1-5 using VLMEvalKit?

In this tutorial, we will provide a detailed guide on how to evaluate Mini-InternVL-Chat-2B-V1-5 using VLMEvalKit.

First of all, please follow this [guide](https://github.com/open-compass/VLMEvalKit/blob/main/Quickstart.md) to install VLMEvalKit.

## MMBench_DEV_EN

```
torchrun --nproc-per-node=8 run.py --data MMBench_DEV_EN --model Mini-InternVL-Chat-2B-V1-5 --verbose
```

The result is:

```
2024-05-24 23:38:26,074 - Evaluation - INFO -   split   Overall  ...  spatial_relationship  structuralized_imagetext_understanding
0   dev  0.706186  ...              0.266667                                0.423077
```

## MMBench_DEV_CN

```
torchrun --nproc-per-node=8 run.py --data MMBench_DEV_CN --model Mini-InternVL-Chat-2B-V1-5 --verbose
```

The result is:

```
2024-05-24 23:38:10,864 - Evaluation - INFO -   split   Overall  ...  spatial_relationship  structuralized_imagetext_understanding
0   dev  0.656357  ...              0.222222                                0.307692
```

## MMStar

```
torchrun --nproc-per-node=8 run.py --data MMStar --model Mini-InternVL-Chat-2B-V1-5 --verbose
```

The result is:

```
2024-05-24 23:38:37,502 - Evaluation - INFO -   split   Overall  ...   math  science & technology
0  none  0.461333  ...  0.448                 0.372
```

## MME

```
torchrun --nproc-per-node=8 run.py --data MME --model Mini-InternVL-Chat-2B-V1-5 --verbose
```

The result is:

```
2024-05-24 23:38:42,360 - Evaluation - INFO -     perception   reasoning    OCR  ...     posters  scene  text_translation
0  1475.888655  423.928571  147.5  ...  130.952381  151.0             170.0
```

## SEEDBench_IMG

```
torchrun --nproc-per-node=8 run.py --data SEEDBench_IMG --model Mini-InternVL-Chat-2B-V1-5 --verbose
```

The result is:

```
2024-05-24 23:39:49,107 - Evaluation - INFO -   split   Overall  ...  Text Understanding  Visual Reasoning
0  none  0.694491  ...            0.690476          0.731118
```

## MMVet

```
torchrun --nproc-per-node=8 run.py --data MMVet --model Mini-InternVL-Chat-2B-V1-5 --verbose
```

The result is:

```
2024-05-24 23:38:53,665 - Evaluation - INFO -   Category  tot        acc
0      rec  187  42.352941
1      ocr  108  42.500000
2     know   84  20.357143
3      gen   80  22.375000
4     spat   75  42.533333
5     math   26  18.461538
6  Overall  218  38.256881
```

Note that because the version of GPT used for scoring differs from the official server, the scores tested by VLMEvalKit will be slightly different.

## MMMU_DEV_VAL

```
torchrun --nproc-per-node=8 run.py --data MMMU_DEV_VAL --model Mini-InternVL-Chat-2B-V1-5 --verbose
```

The result is:

```
2024-05-24 23:39:23,721 - Evaluation - INFO -         split   Overall  ...   Science  Tech & Engineering
0         dev  0.353333  ...  0.240000            0.342857
1  validation  0.376667  ...  0.286667            0.376190
```

## MathVista_MINI

```
torchrun --nproc-per-node=8 run.py --data MathVista_MINI --model Mini-InternVL-Chat-2B-V1-5 --verbose
```

The result is:

```
0                       Overall  1000       520  411      52.000000  41.100000
1          scientific reasoning   122        91   54      74.590164  44.262295
2   textbook question answering   158       100   72      63.291139  45.569620
3           numeric commonsense   144        41   45      28.472222  31.250000
4          arithmetic reasoning   353       108  129      30.594901  36.543909
5     visual question answering   179        94   69      52.513966  38.547486
6            geometry reasoning   239       158   85      66.108787  35.564854
7           algebraic reasoning   281       180  104      64.056940  37.010676
8      geometry problem solving   208       149   76      71.634615  36.538462
9             math word problem   186        27   68      14.516129  36.559140
10            logical reasoning    37        24    4      64.864865  10.810811
11    figure question answering   269       150  126      55.762082  46.840149
12        statistical reasoning   301       139  159      46.179402  52.823920
```

Note that because the version of GPT used for answer extraction differs from the official code, the scores tested by VLMEvalKit will be slightly different.

## ScienceQA_TEST

```
torchrun --nproc-per-node=8 run.py --data ScienceQA_TEST --model Mini-InternVL-Chat-2B-V1-5 --verbose
```

The result is:

```
2024-05-24 23:42:24,271 - Evaluation - INFO -   split   Overall  ...  Weather and climate  World religions
0  test  0.852256  ...             0.810345              0.0
```

## HallusionBench

```
torchrun --nproc-per-node=8 run.py --data HallusionBench --model Mini-InternVL-Chat-2B-V1-5 --verbose
```

The result is:

```
2024-05-24 23:41:40,703 - Evaluation - INFO -           split       aAcc       fAcc       qAcc
0       Overall  59.411146  24.277457  28.791209
1            VS  61.111111  17.241379  34.269663
2            VD  58.375635  27.826087  25.270758
3   VD_illusion  57.638889  22.580645  19.444444
4        VS_ocr  53.703704  15.384615  11.111111
5        VS_map  54.687500   9.090909  15.625000
6      VS_chart  66.153846  15.000000  50.000000
7      VS_table  62.500000  28.571429  34.883721
8       VD_math  58.333333  11.111111  31.481481
9      VD_video  47.058824  12.500000   8.695652
10    VD_figure  65.000000  41.463415  28.205128
11       VD_ocr  75.280899  53.488372  51.162791
```

## TextVQA_VAL

```
torchrun --nproc-per-node=8 run.py --data TextVQA_VAL --model Mini-InternVL-Chat-2B-V1-5 --verbose
```

The result is:

```
2024-05-24 23:45:09,563 - Evaluation - INFO -    Overall
0   70.452
```

## ChartQA_TEST

```
torchrun --nproc-per-node=8 run.py --data ChartQA_TEST --model Mini-InternVL-Chat-2B-V1-5 --verbose
```

The result is:

```
2024-05-24 23:43:24,645 - Evaluation - INFO -    test_augmented  test_human  Overall
0           91.68       54.88    73.28
```

## AI2D_TEST

```
torchrun --nproc-per-node=8 run.py --data AI2D_TEST --model Mini-InternVL-Chat-2B-V1-5 --verbose
```

The result is:

```
2024-05-24 23:43:12,535 - Evaluation - INFO -   split   Overall  atomStructure  ...  typesOf  volcano  waterCNPCycle
0  none  0.699482          0.625  ...  0.61516   0.6875       0.477273
```

## LLaVABench

```
torchrun --nproc-per-node=8 run.py --data LLaVABench --model Mini-InternVL-Chat-2B-V1-5 --verbose
```

The result is:

```
Processing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:00 60/60 100%
     split  Relative Score (main)  VLM Score  GPT4 Score
0  overall                   61.0       47.7        78.2
1  complex                   68.4       52.5        76.8
2     conv                   59.9       53.5        89.4
3   detail                   47.1       32.0        68.0
```

## DocVQA_VAL

```
torchrun --nproc-per-node=8 run.py --data DocVQA_VAL --model Mini-InternVL-Chat-2B-V1-5 --verbose
```

The result is:

```
2024-05-24 23:47:40,385 - Evaluation - INFO -          val    Overall
0  83.883006  83.883006
```

## InfoVQA_VAL

```
torchrun --nproc-per-node=8 run.py --data InfoVQA_VAL --model Mini-InternVL-Chat-2B-V1-5 --verbose
```

The result is:

```
2024-05-24 23:46:53,303 - Evaluation - INFO -         val   Overall
0  55.86691  55.86691
```

## OCRBench

```
torchrun --nproc-per-node=8 run.py --data OCRBench --model Mini-InternVL-Chat-2B-V1-5 --verbose
```

The result is:

```
2024-05-24 23:45:30,929 - Evaluation - INFO - Score:
2024-05-24 23:45:30,929 - Evaluation - INFO - Text Recognition:222
2024-05-24 23:45:30,929 - Evaluation - INFO - Scene Text-centric VQA:163
2024-05-24 23:45:30,929 - Evaluation - INFO - Doc-oriented VQA:125
2024-05-24 23:45:30,929 - Evaluation - INFO - Key Information Extraction:139
2024-05-24 23:45:30,929 - Evaluation - INFO - Handwritten Mathematical Expression Recognition:5
2024-05-24 23:45:30,929 - Evaluation - INFO - Final Score:654
2024-05-24 23:45:30,929 - Evaluation - INFO - Final Score Norm:65.4
```

## RealWorldQA

```
torchrun --nproc-per-node=8 run.py --data RealWorldQA --model Mini-InternVL-Chat-2B-V1-5 --verbose
```

The result is:

```
2024-05-25 00:13:34,645 - Evaluation - INFO -   split   Overall
0  none  0.579085
```

## SEEDBench2_Plus

```
torchrun --nproc-per-node=8 run.py --data SEEDBench2_Plus --model Mini-InternVL-Chat-2B-V1-5 --verbose
```

The result is:

```
2024-05-29 12:31:50,587 - Evaluation - INFO -   split   Overall     chart       map       web
0  none  0.588933  0.562963  0.482032  0.751515
```
