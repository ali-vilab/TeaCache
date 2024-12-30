# Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model

<div class="is-size-5 publication-authors", align="center",>
            <span class="author-block">
              <a href="https://liewfeng.github.io" target="_blank">Feng Liu</a><sup>1</sup><sup>*</sup>,&nbsp;
            </span>
            <span class="author-block">
              <a href="https://scholar.google.com.hk/citations?user=ZO3OQ-8AAAAJ" target="_blank">Shiwei Zhang</a><sup>2</sup>,&nbsp;
            </span>
            <span class="author-block">
              <a href="https://jeffwang987.github.io" target="_blank">Xiaofeng Wang</a><sup>1,3</sup>,&nbsp;
            </span>
            <span class="author-block">
              <a href="https://weilllllls.github.io" target="_blank">Yujie Wei</a><sup>4</sup>,&nbsp;
            </span>
            <span class="author-block">
              <a href="http://haonanqiu.com" target="_blank">Haonan Qiu</a><sup>5</sup>
            </span>
            <br>
            <span class="author-block">
              <a href="https://callsys.github.io/zhaoyuzhong.github.io-main" target="_blank">Yuzhong Zhao</a><sup>1</sup>,&nbsp;
            </span>
            <span class="author-block">
              <a href="https://scholar.google.com.sg/citations?user=16RDSEUAAAAJ" target="_blank">Yingya Zhang</a><sup>2</sup>,&nbsp;
            </span>
            <span class="author-block">
              <a href="https://scholar.google.com/citations?user=tjEfgsEAAAAJ&hl=en&oi=ao" target="_blank">Qixiang Ye</a><sup>1</sup>,&nbsp;
            </span>
            <span class="author-block">
              <a href="https://scholar.google.com/citations?user=0IKavloAAAAJ&hl=en&oi=ao" target="_blank">Fang Wan</a><sup>1</sup><sup>†</sup>
            </span>
          </div>

<div class="is-size-5 publication-authors", align="center">
            <span class="author-block"><sup>1</sup>University of Chinese Academy of Sciences,&nbsp;</span>
            <span class="author-block"><sup>2</sup>Alibaba Group</span>
            <br>
            <span class="author-block"><sup>3</sup>Institute of Automation, Chinese Academy of Sciences</span>
            <br>
            <span class="author-block"><sup>4</sup>Fudan University,&nbsp;</span>
            <span class="author-block"><sup>5</sup>Nanyang Technological University</span>
          </div>


<div class="is-size-5 publication-authors", align="center">
            (* Work was done during internship at Alibaba Group. † Corresponding author.)
          </div>

<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.  </h2>

<h5 align="center">

[![hf_paper](https://img.shields.io/badge/🤗-Paper%20In%20HF-red.svg)](https://huggingface.co/papers/2411.19108)
[![arXiv](https://img.shields.io/badge/Arxiv-2411.19108-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2411.19108) 
[![Home Page](https://img.shields.io/badge/Project-<Website>-blue.svg)](https://liewfeng.github.io/TeaCache/) 
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](./LICENSE) 
[![github](https://img.shields.io/github/stars/LiewFeng/TeaCache.svg?style=social)](https://github.com/LiewFeng/TeaCache/)

</h5>



![visualization](./assets/tisser.png)

## Latest News 🔥
- **Welcome for PRs to support other models. Please star ⭐ our project and stay tuned.**
- [2024/12/30] 🔥 Support [Mochi](https://github.com/genmoai/mochi) and [LTX-Video](https://github.com/Lightricks/LTX-Video) for Video Diffusion Models. Support [Lumina-T2X](https://github.com/Alpha-VLLM/Lumina-T2X) for Image Diffusion Models.
- [2024/12/27] 🔥 Support [FLUX](https://github.com/black-forest-labs/flux). TeaCache works well for Image Diffusion Models!
- [2024/12/26] 🔥 Support [ConsisID](https://github.com/PKU-YuanGroup/ConsisID). Thanks [@SHYuanBest](https://github.com/SHYuanBest). TeaCache can be easily adapted to models based on CogvideoX.
- [2024/12/24] 🔥 Support [HunyuanVideo](https://github.com/Tencent/HunyuanVideo).
- [2024/12/19] 🔥 Support [CogVideoX](https://github.com/THUDM/CogVideo).
- [2024/12/06] 🎉 Release the [code](https://github.com/LiewFeng/TeaCache) of TeaCache. Support [Open-Sora](https://github.com/hpcaitech/Open-Sora), [Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan) and [Latte](https://github.com/Vchitect/Latte).
- [2024/11/28] 🎉 Release the [paper](https://arxiv.org/abs/2411.19108) of TeaCache.

## Introduction
We introduce Timestep Embedding Aware Cache (TeaCache), a training-free caching approach that estimates and leverages the fluctuating differences among model outputs across timesteps, thereby accelerating the inference. For more details and visual results, please visit our [project page](https://github.com/LiewFeng/TeaCache).

## TeaCache for HunyuanVideo
Please refer to [TeaCache4HunyuanVideo](./TeaCache4HunyuanVideo/README.md).

## TeaCache for ConsisID

Please refer to [TeaCache4ConsisID](./TeaCache4ConsisID/README.md).

## TeaCache for FLUX

Please refer to [TeaCache4FLUX](./TeaCache4FLUX/README.md).

## TeaCache for Mochi

Please refer to [TeaCache4Mochi](./TeaCache4Mochi/README.md).

## TeaCache for LTX-Video

Please refer to [TeaCache4LTX-Video](./TeaCache4LTX-Video/README.md).

## TeaCache for Lumina-T2X

Please refer to [TeaCache4Lumina-T2X](./TeaCache4Lumina-T2X/README.md).

## Installation

Prerequisites:

- Python >= 3.10
- PyTorch >= 1.13 (We recommend to use a >2.0 version)
- CUDA >= 11.6

We strongly recommend using Anaconda to create a new environment (Python >= 3.10) to run our examples:

```shell
conda create -n teacache python=3.10 -y
conda activate teacache
```

Install TeaCache:

```shell
git clone https://github.com/LiewFeng/TeaCache
cd TeaCache
pip install -e .
```


## Evaluation of TeaCache

We first generate videos according to VBench's prompts.

And then calculate Vbench, PSNR, LPIPS and SSIM based on the video generated.

1. Generate video
```
cd eval/teacache
python experiments/latte.py
python experiments/opensora.py
python experiments/open_sora_plan.py
python experiments/cogvideox.py
```

2. Calculate Vbench score
```
# vbench is calculated independently
# get scores for all metrics
python vbench/run_vbench.py --video_path aaa --save_path bbb
# calculate final score
python vbench/cal_vbench.py --score_dir bbb
```

3. Calculate other metrics
```
# these metrics are calculated compared with original model
# gt video is the video of original model
# generated video is our methods's results
python common_metrics/eval.py --gt_video_dir aa --generated_video_dir bb
```
## Acknowledgement

This repository is built based on [VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys), [Diffusers](https://github.com/huggingface/diffusers), [Open-Sora](https://github.com/hpcaitech/Open-Sora), [Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan), [Latte](https://github.com/Vchitect/Latte), [CogVideoX](https://github.com/THUDM/CogVideo), [HunyuanVideo](https://github.com/Tencent/HunyuanVideo), [ConsisID](https://github.com/PKU-YuanGroup/ConsisID), [FLUX](https://github.com/black-forest-labs/flux), [Mochi](https://github.com/genmoai/mochi), [LTX-Video](https://github.com/Lightricks/LTX-Video) and [Lumina-T2X](https://github.com/Alpha-VLLM/Lumina-T2X). Thanks for their contributions!

## License

* The majority of this project is released under the Apache 2.0 license as found in the [LICENSE](./LICENSE) file.
* For [VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys), [Diffusers](https://github.com/huggingface/diffusers), [Open-Sora](https://github.com/hpcaitech/Open-Sora), [Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan), [Latte](https://github.com/Vchitect/Latte), [CogVideoX](https://github.com/THUDM/CogVideo), [HunyuanVideo](https://github.com/Tencent/HunyuanVideo), [ConsisID](https://github.com/PKU-YuanGroup/ConsisID), [FLUX](https://github.com/black-forest-labs/flux), [Mochi](https://github.com/genmoai/mochi), [LTX-Video](https://github.com/Lightricks/LTX-Video) and [Lumina-T2X](https://github.com/Alpha-VLLM/Lumina-T2X), please follow their LICENSE.
* The service is a research preview. Please contact us if you find any potential violations. (liufeng20@mails.ucas.ac.cn)

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


