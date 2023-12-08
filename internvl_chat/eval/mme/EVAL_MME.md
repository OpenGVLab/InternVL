# MME Benchmark

[MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) is a comprehensive evaluation benchmark for multimodal large language models. It measures both perception and cognition abilities on a total of 14 subtasks, including existence, count, position, color, poster, celebrity, scene, landmark, artwork, OCR, commonsense reasoning, numerical calculation, text translation, and code reasoning.

Qwen-VL-Chat achieves SOTAs on both perception and cognition evaluation.

Perception Evaluation

| Rank |                         Model                          |                     Version                      |    Score    |
| :--: | :----------------------------------------------------: | :----------------------------------------------: | :---------: |
|  1   | **[Qwen-VL-Chat](https://github.com/QwenLM/Qwen-VL/)** | **[Qwen-7B](https://github.com/QwenLM/Qwen-7B)** | **1487.57** |
|  2   |                       Skywork-MM                       |                  Skywork-MM-13B                  |   1419.08   |
|  3   |                         MMICL                          |                    FlanT5xxl                     |   1376.00   |
|  4   |                          Lynx                          |                    vicuna-7b                     |   1373.23   |
|  5   |                         BLIVA                          |                    FlanT5xxl                     |   1337.73   |

Cognition Evaluation

| Rank |                         Model                          |                     Version                      |   Score    |
| :--: | :----------------------------------------------------: | :----------------------------------------------: | :--------: |
|  1   | **[Qwen-VL-Chat](https://github.com/QwenLM/Qwen-VL/)** | **[Qwen-7B](https://github.com/QwenLM/Qwen-7B)** | **360.71** |
|  2   |                         MMICL                          |                    FlanT5xxl                     |   360.36   |
|  3   |                       Skywork-MM                       |                  Skywork-MM-13B                  |   356.43   |
|  4   |                         BLIVA                          |                    FlanT5xxl                     |   331.43   |
|  5   |                    LRV-Instruction                     |                      LRV-7B                      |   328.21   |

Full Metrics

```
=========== Perception ===========
total score: 1487.576330532213

         existence  score: 158.33333333333331
         count  score: 150.0
         position  score: 128.33333333333334
         color  score: 170.0
         posters  score: 178.57142857142856
         celebrity  score: 120.58823529411764
         scene  score: 152.25
         landmark  score: 164.0
         artwork  score: 125.5
         OCR  score: 140.0


=========== Cognition ===========
total score: 360.71428571428567

         commonsense_reasoning  score: 130.7142857142857
         numerical_calculation  score: 40.0
         text_translation  score: 147.5
         code_reasoning  score: 42.5
```

## How To Reproduce Results of MME Benchmark

1. Download MME images and eval_tool from the [MME repo](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/blob/Evaluation/README.md)
2. Rearrange images by executing `python get_images.py`
3. Evaluate Qwen-VL-Chat results by executing `python eval.py`
4. Calculate MME results by executing `python calculation.py --results_dir Qwen-VL-Chat`, which the calculation script comes from the MME eval_tool.
