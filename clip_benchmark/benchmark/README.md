# Benchmark

the benchmark results are available in [benchmark.csv](benchmark.csv).
You can visualize the results in the [notebook](results.ipynb)

# How to reproduce the CLIP benchmark results

## Webdataset evaluation: VTAB+ and retrieval datasets (MSCOCO, Flickr8k, Flickr30k)

```bash
clip_benchmark eval --pretrained_model openai openclip_base \
    --dataset "webdatasets.txt" \
    --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" \
    --output "benchmark_{dataset}_{pretrained}_{model}_{language}_{task}.json"
```

Once the evaluation finishes, you can construct a CSV with all the results:

```bash
clip_benchmark build benchmark_*.json --output benchmark.csv
```

*Notes:* Pascal VOC 2007 multilabel is not yet included in the webdataset test suite. Multilingual support with webdataset is in progress.

## Alternative: Local download

```bash
clip_benchmark eval --pretrained_model  openai openclip_base  --dataset vtab+ retrieval \
--dataset_root "clip_benchmark_datasets/{dataset}" \
--output "benchmark_{dataset}_{pretrained}_{model}_{language}_{task}.json"
```

(Change `--dataset_root` accordingly)

## Multilingual ImageNet benchmark

To run the multilingual ImageNet benchmark, use:

```bash
clip_benchmark eval --pretrained_model openclip_multilingual openclip_base openai  --dataset imagenet1k --language cn it jp en ar\
--dataset_root "clip_benchmark_datasets/{dataset}" \
--output "multilingual_{dataset}_{pretrained}_{model}_{language}_{task}.json"
```

(Change `--dataset_root` accordingly)

## Multilingual MS-COCO benchmark

To run the multilingual MS-COCO benchmark, use:

```bash
clip_benchmark eval --pretrained_model openclip_multilingual openclip_base openai --dataset multilingual_mscoco_captions --language es it ko pl ru tr zh en \
--dataset_root "clip_benchmark_datasets/{dataset}" \
--output "multilingual_{dataset}_{pretrained}_{model}_{language}_{task}.json"
```

(Change `--dataset_root` accordingly)
