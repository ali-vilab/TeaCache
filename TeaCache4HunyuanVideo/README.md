<!-- ## **TeaCache4HunyuanVideo** -->
# TeaCache4HunyuanVideo

[TeaCache](https://github.com/LiewFeng/TeaCache) can speedup [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) 2x without much visual quality degradation, in a training-free manner. The following video shows the results generated by TeaCache-HunyuanVideo with various `rel_l1_thresh` values: 0 (original), 0.1 (1.6x speedup), 0.15 (2.1x speedup).

https://github.com/user-attachments/assets/34b5dab0-5b0f-48a0-968d-88af18b84803


## 📈 Inference Latency Comparisons on a Single A800 GPU


|      Resolution       |        HunyuanVideo       |    TeaCache (0.1)    |     TeaCache (0.15)    |
|:---------------------:|:-------------------------:|:--------------------:|:----------------------:|
|         540p          |        ~18 min            |     ~11 min          |       ~8 min           |
|         720p          |        ~50 min            |     ~30 min          |       ~23 min          | 


## Usage

Follow [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) to clone the repo and finish the installation, then copy 'teacache_sample_video.py' in this repo to the HunyuanVideo repo. You can modify the '`rel_l1_thresh`' in line 220 to obtain your desired trade-off between latency and visul quality.

For single-gpu inference, you can use the following command:

```bash
cd HunyuanVideo

python3 teacache_sample_video.py \
    --video-size 720 1280 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --use-cpu-offload \
    --save-path ./teacache_results
```

To generate a video with 8 GPUs, you can use the following command:

```bash
cd HunyuanVideo

torchrun --nproc_per_node=8 teacache_sample_video.py \
    --video-size 1280 720 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --seed 42 \
    --ulysses-degree 8 \
    --ring-degree 1 \
    --save-path ./teacache_results
```

For FP8 inference, you must explicitly specify the FP8 weight path. For example, to generate a video with fp8 weights, you can use the following command:

```bash
cd HunyuanVideo

DIT_CKPT_PATH={PATH_TO_FP8_WEIGHTS}/{WEIGHT_NAME}_fp8.pt

python3 teacache_sample_video.py \
    --dit-weight ${DIT_CKPT_PATH} \
    --video-size 1280 720 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --seed 42 \
    --embedded-cfg-scale 6.0 \
    --flow-shift 7.0 \
    --flow-reverse \
    --use-cpu-offload \
    --use-fp8 \
    --save-path ./teacache_fp8_results
```

## Citation
If you find TeaCache is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.

```
@article{liu2024timestep,
  title={Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model},
  author={Liu, Feng and Zhang, Shiwei and Wang, Xiaofeng and Wei, Yujie and Qiu, Haonan and Zhao, Yuzhong and Zhang, Yingya and Ye, Qixiang and Wan, Fang},
  journal={arXiv preprint arXiv:2411.19108},
  year={2024}
}
```


## Acknowledgements

We would like to thank the contributors to the [HunyuanVideo](https://github.com/Tencent/HunyuanVideo).
