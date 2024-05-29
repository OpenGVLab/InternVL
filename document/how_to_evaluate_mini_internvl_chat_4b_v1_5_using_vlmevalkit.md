# How to Evaluate Mini-InternVL-Chat-4B-V1-5 using VLMEvalKit?

In this tutorial, we will provide a detailed guide on how to evaluate Mini-InternVL-Chat-4B-V1-5 using VLMEvalKit.

First of all, please follow this [guide](https://github.com/open-compass/VLMEvalKit/blob/main/Quickstart.md) to install VLMEvalKit.

## MMBench_DEV_EN

```
torchrun --nproc-per-node=8 run.py --data MMBench_DEV_EN --model Mini-InternVL-Chat-4B-V1-5 --verbose
```

The result is:

```
2024-05-29 04:01:07,750 - Evaluation - INFO -   split   Overall  ...  spatial_relationship  structuralized_imagetext_understanding
0   dev  0.764605  ...              0.355556                                0.551282
```

## MMBench_DEV_CN

```
torchrun --nproc-per-node=8 run.py --data MMBench_DEV_CN --model Mini-InternVL-Chat-4B-V1-5 --verbose
```

The result is:

```
2024-05-29 04:11:49,747 - Evaluation - INFO -   split   Overall  ...  spatial_relationship  structuralized_imagetext_understanding
0   dev  0.699313  ...              0.244444                                0.512821
```

## MMStar

```
torchrun --nproc-per-node=8 run.py --data MMStar --model Mini-InternVL-Chat-4B-V1-5 --verbose
```

The result is:

```
2024-05-29 04:02:01,943 - Evaluation - INFO -   split   Overall  ...   math  science & technology
0  none  0.527333  ...  0.516                 0.408
```

## MME

```
torchrun --nproc-per-node=8 run.py --data MME --model Mini-InternVL-Chat-4B-V1-5 --verbose
```

The result is:

```
2024-05-29 04:06:31,735 - Evaluation - INFO -     perception   reasoning    OCR  ...     posters   scene  text_translation
0  1569.933774  492.857143  147.5  ...  153.061224  153.25             162.5
```

## SEEDBench_IMG

```
torchrun --nproc-per-node=8 run.py --data SEEDBench_IMG --model Mini-InternVL-Chat-4B-V1-5 --verbose
```

The result is:

```
2024-05-29 04:17:50,620 - Evaluation - INFO -   split   Overall  ...  Text Understanding  Visual Reasoning
0  none  0.721684  ...            0.559524          0.779456
```

## MMVet

```
torchrun --nproc-per-node=8 run.py --data MMVet --model Mini-InternVL-Chat-4B-V1-5 --verbose
```

The result is:

```
2024-05-29 11:04:32,615 - Evaluation - INFO -   Category  tot        acc
0      rec  187  43.636364
1      ocr  108  47.037037
2     know   84  26.904762
3      gen   80  27.625000
4     spat   75  43.066667
5     math   26  34.230769
6  Overall  218  41.972477
```

Note that because the version of GPT used for scoring differs from the official server, the scores tested by VLMEvalKit will be slightly different.

## MMMU_DEV_VAL

```
torchrun --nproc-per-node=8 run.py --data MMMU_DEV_VAL --model Mini-InternVL-Chat-4B-V1-5 --verbose
```

The result is:

```
2024-05-29 04:16:48,300 - Evaluation - INFO -         split   Overall  ...  Science  Tech & Engineering
0  validation  0.457778  ...      0.4            0.404762
1         dev  0.480000  ...      0.4            0.371429
```

## MathVista_MINI

```
torchrun --nproc-per-node=8 run.py --data MathVista_MINI --model Mini-InternVL-Chat-4B-V1-5 --verbose
```

The result is:

```
2024-05-29 04:56:36,722 - Evaluation - INFO -                      Task&Skill   tot  prefetch  hit  prefetch_rate        acc
0                       Overall  1000       560  537      56.000000  53.700000
1          scientific reasoning   122        86   71      70.491803  58.196721
2   textbook question answering   158        99   89      62.658228  56.329114
3           numeric commonsense   144        41   41      28.472222  28.472222
4          arithmetic reasoning   353       134  180      37.960340  50.991501
5     visual question answering   179        94   88      52.513966  49.162011
6            geometry reasoning   239       161  118      67.364017  49.372385
7           algebraic reasoning   281       189  139      67.259786  49.466192
8      geometry problem solving   208       153  105      73.557692  50.480769
9             math word problem   186        52   96      27.956989  51.612903
10            logical reasoning    37        18    4      48.648649  10.810811
11    figure question answering   269       162  159      60.223048  59.107807
12        statistical reasoning   301       177  203      58.803987  67.441860
```

Note that because the version of GPT used for answer extraction differs from the official code, the scores tested by VLMEvalKit will be slightly different.

## ScienceQA_TEST

```
torchrun --nproc-per-node=8 run.py --data ScienceQA_TEST --model Mini-InternVL-Chat-4B-V1-5 --verbose
```

The result is:

```
2024-05-29 04:14:56,970 - Evaluation - INFO -   split   Overall  ...  Weather and climate  World religions
0  test  0.927119  ...             0.948276              1.0
```

## HallusionBench

```
torchrun --nproc-per-node=8 run.py --data HallusionBench --model Mini-InternVL-Chat-4B-V1-5 --verbose
```

The result is:

```
2024-05-29 04:24:07,079 - Evaluation - INFO -           split       aAcc       fAcc       qAcc
0       Overall  62.460568  33.236994  32.747253
1            VD  58.544839  31.739130  24.187726
2            VS  68.888889  36.206897  46.067416
3   VD_illusion  59.027778  30.645161  23.611111
4       VD_math  55.555556  25.000000  27.777778
5        VS_ocr  61.111111  34.615385  25.925926
6     VD_figure  62.500000  41.463415  23.076923
7        VD_ocr  74.157303  51.162791  48.837209
8        VS_map  54.687500  27.272727  15.625000
9      VS_chart  76.153846  40.000000  64.473684
10     VS_table  72.321429  39.285714  48.837209
11     VD_video  50.000000  12.500000   7.246377
```

## TextVQA_VAL

```
torchrun --nproc-per-node=8 run.py --data TextVQA_VAL --model Mini-InternVL-Chat-4B-V1-5 --verbose
```

The result is:

```
2024-05-29 04:26:03,441 - Evaluation - INFO -    Overall
0   72.886
```

## ChartQA_TEST

```
torchrun --nproc-per-node=8 run.py --data ChartQA_TEST --model Mini-InternVL-Chat-4B-V1-5 --verbose
```

The result is:

```
2024-05-29 04:28:40,199 - Evaluation - INFO -    test_augmented  test_human  Overall
0            93.2        68.8     81.0
```

## AI2D_TEST

```
torchrun --nproc-per-node=8 run.py --data AI2D_TEST --model Mini-InternVL-Chat-4B-V1-5 --verbose
```

The result is:

```
2024-05-29 04:29:54,674 - Evaluation - INFO -   split   Overall  atomStructure  ...   typesOf  volcano  waterCNPCycle
0  none  0.769754          0.875  ...  0.720117    0.875            0.5
```

## LLaVABench

```
torchrun --nproc-per-node=8 run.py --data LLaVABench --model Mini-InternVL-Chat-4B-V1-5 --verbose
```

The result is:

```
Processing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:00 60/60 100%
     split  Relative Score (main)  VLM Score  GPT4 Score
0  overall                   68.3       49.7        72.7
1  complex                   77.5       55.4        71.4
2   detail                   67.0       40.7        60.7
3     conv                   56.6       48.2        85.3
```

## DocVQA_VAL

```
torchrun --nproc-per-node=8 run.py --data DocVQA_VAL --model Mini-InternVL-Chat-4B-V1-5 --verbose
```

The result is:

```
2024-05-29 04:40:18,175 - Evaluation - INFO -          val    Overall
0  86.635414  86.635414
```

## InfoVQA_VAL

```
torchrun --nproc-per-node=8 run.py --data InfoVQA_VAL --model Mini-InternVL-Chat-4B-V1-5 --verbose
```

The result is:

```
2024-05-29 04:49:18,835 - Evaluation - INFO -          val    Overall
0  64.588708  64.588708
```

## OCRBench

```
torchrun --nproc-per-node=8 run.py --data OCRBench --model Mini-InternVL-Chat-4B-V1-5 --verbose
```

The result is:

```
2024-05-29 04:31:09,980 - Evaluation - INFO - Score:
2024-05-29 04:31:09,980 - Evaluation - INFO - Text Recognition:194
2024-05-29 04:31:09,980 - Evaluation - INFO - Scene Text-centric VQA:160
2024-05-29 04:31:09,980 - Evaluation - INFO - Doc-oriented VQA:145
2024-05-29 04:31:09,980 - Evaluation - INFO - Key Information Extraction:133
2024-05-29 04:31:09,980 - Evaluation - INFO - Handwritten Mathematical Expression Recognition:6
2024-05-29 04:31:09,980 - Evaluation - INFO - Final Score:638
2024-05-29 04:31:09,980 - Evaluation - INFO - Final Score Norm:63.8
```

## RealWorldQA

```
torchrun --nproc-per-node=8 run.py --data RealWorldQA --model Mini-InternVL-Chat-4B-V1-5 --verbose
```

The result is:

```
2024-05-29 04:34:10,091 - Evaluation - INFO -   split   Overall
0  none  0.601307
```

## SEEDBench2_Plus

```
torchrun --nproc-per-node=8 run.py --data SEEDBench2_Plus --model Mini-InternVL-Chat-4B-V1-5 --verbose
```

The result is:

```
2024-05-29 12:33:20,074 - Evaluation - INFO -   split   Overall     chart       map       web
0  none  0.625823  0.616049  0.537794  0.745455
```
