## Generative Semantic Communication: Diffusion Models Beyond Bit Recovery
### [Eleonora Grassucci](https://sites.google.com/uniroma1.it/eleonoragrassucci/home-page), [Sergio Barbarossa](https://sites.google.com/a/uniroma1.it/sergiobarbarossa/), and [Danilo Comminiello](https://danilocomminiello.site.uniroma1.it/)

[[ArXiv Preprint](https://arxiv.org/abs/2306.04321v1)]

This repository is under construction! :)

### :page_with_curl: Abstract
Semantic communication is expected to be one of the cores of next-generation AI-based communications. One of the possibilities offered by semantic communication is the capability to regenerate, at the destination side, images or videos semantically equivalent to the transmitted ones, without necessarily recovering the transmitted sequence of bits. The current solutions still lack the ability to build complex scenes from the received partial information. Clearly, there is an unmet need to balance the effectiveness of generation methods and the complexity of the transmitted information, possibly taking into account the goal of communication. In this paper, we aim to bridge this gap by proposing a novel generative diffusion-guided framework for semantic communication that leverages the strong abilities of diffusion models in synthesizing multimedia content while preserving semantic features. We reduce bandwidth usage by sending highly-compressed semantic information only. Then, the diffusion model learns to synthesize semantic-consistent scenes through spatially-adaptive normalizations from such denoised semantic information.
We prove, through an in-depth assessment of multiple scenarios,  that our method outperforms existing solutions in generating high-quality images with preserved semantic information even in cases where the received content is significantly degraded. More specifically, our results show that objects, locations, and depths are still recognizable even in the presence of extremely noisy conditions of the communication channel.

### :dart: The GESCO framework
<img src="architecture-Pagina-1.drawio.png"/>

### :chart_with_upwards_trend: Main Results

<img src="fig1-Pagina-1.drawio.png"/>

### :clipboard: How to use GESCO

#### Train GESCO

* Install the requirements.

* Run the following command:

```
python image_train.py --data_dir ./data --dataset_mode cityscapes --lr 1e-4 --batch_size 4 --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True
--noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --use_checkpoint True --num_classes 35
--class_cond True --no_instance False 
```

For Cityscapes: `--dataset_mode cityscapes`, `--image_size 256`, `--num_classes 35`, `--class_cond True`, `--no_instance False`.

For COCO: `--dataset_mode coco`, `--image_size 256`, `--num_classes 183`, `--class_cond True`, `--no_instance False`.

For ADE20K: `--dataset_mode ade20k`, `--image_size 256`, `--num_classes 151`, `--class_cond True`, `--no_instance True`.

#### Sample from GESCO

* Run the following command:

```
python image_sample.py --data_dir "./data" --dataset_mode cityscapes --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --num_classes 35 --class_cond True --no_instance False --batch_size 1 --num_samples 100 --model_path ./your_checkpoint_path.pt --results_path ./your_results_path --s 2 --one_hot_label True --snr your_snr_value --pool None --unet_model unet 
```

With the same dataset-specific hyperparameters, in addition to `--s` with is equal to `2` in Cityscapes and `2.5` for COCO and ADE20k.

Our code is based on [guided-diffusion](https://github.com/openai/guided-diffusion) and on [SDM](https://github.com/WeilunWang/semantic-diffusion-model).

#### Cite
Please, cite our work if you found it useful.

```
@article{grassucci2023generative,
    title={Generative Semantic Communication: Diffusion Models Beyond Bit Recovery},
    author={Grassucci, Eleonora and Barbarossa, Sergio and Comminiello, Danilo},
    year={2023},
    journal={ArXiv preprint: ArXiv:2306.04321},
}
```
