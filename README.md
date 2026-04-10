# Object-Shot Enhanced Grounding Network for Egocentric Video (*CVPR 2025*)

> Official implementation of OSGNet at CVPR 2025, and the champion solution repository for three egocentric video localization tracks of the Ego4D Episodic Memory Challenge at CVPR 2025.

## Authors

**Yisen Feng**<sup>1</sup>, **Haoyu Zhang**<sup>1,2</sup>, **Meng Liu**<sup>3</sup>\*, **Weili Guan**<sup>1</sup>, **Liqiang Nie**<sup>1</sup>\*

<sup>1</sup> Harbin Institute of Technology (Shenzhen) <sup>2</sup> Pengcheng Laboratory <sup>3</sup> Shandong Jianzhu University

\* represents corresponding author

## Links

* **Paper**: [Object-Shot Enhanced Grounding Network for Egocentric Video](https://openaccess.thecvf.com/content/CVPR2025/html/Feng_Object-Shot_Enhanced_Grounding_Network_for_Egocentric_Video_CVPR_2025_paper.html)
* **Technical Report**: [OSGNet@ Ego4D Episodic Memory Challenge 2025](https://arxiv.org/abs/2506.03710)
* **Code Repository**: [iLearn-Lab/CVPR25-OSGNet](https://github.com/iLearn-Lab/CVPR25-OSGNet)
* **Hugging Face**: [iLearn-Lab/CVPR25-OSGNet](https://huggingface.co/iLearn-Lab/CVPR25-OSGNet)
---

## Table of Contents

* [Introduction](#introduction)
* [Highlights](#highlights)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Checkpoints / Models](#checkpoints--models)
* [Dataset / Benchmark](#dataset--benchmark)
* [Usage](#usage)
* [Citation](#citation)
* [Acknowledgement](#acknowledgement)
* [License](#license)

---

## Introduction

This repo is the official implementation of **OSGNet** at CVPR 2025. It is also the champion solution repository for three egocentric video localization tracks of the Ego4D Episodic Memory Challenge at CVPR 2025.

This repo supports **data pre-processing**, **training**, and **evaluation** for the following datasets:

* **Ego4D-NLQ**
* **Ego4D-GoalStep**
* **TACoS**

The repository currently provides:

* training scripts
* inference / evaluation scripts
* feature preparation instructions
* pretrained checkpoints

---

## Highlights

* Supports egocentric and video grounding benchmarks including **Ego4D-NLQ**, **Ego4D-GoalStep**, and **TACoS**
* Provides scripts for **pretraining**, **finetuning**, and **inference**
* Includes instructions for preparing **text features**, **video features**, **LaViLa captions**, and **object features**
* Releases checkpoints for multiple settings

---

## Project Structure

```text
.
├── configs/       # Configuration files for different datasets
├── ego4d_data/    # Annotation files and dataset-related metadata
├── install/       # Environment setup scripts
├── libs/          # Core modules, datasets, modeling, and utilities
├── tools/         # Training scripts
├── train.py
├── eval_nlq.py
├── README.md
└── LICENSE
```

---

## Installation

### 1. Clone the repository and install dependencies

```bash
git clone https://github.com/iLearn-Lab/CVPR25-OSGNet.git
cd CVPR25-OSGNet
```
Follow INSTALL.sh for installing
Recommended PyTorch version:

```text
Torch >= 1.8.0
```

### 2. Prepare offline data

Required resources include:

* text feature
* video feature
* lavila caption (need to unzip)
* object feature



---

## Checkpoints / Models

Download from [huggingface](https://huggingface.co/iLearn-Lab/CVPR25-OSGNet) or from baidu netdisk below.

### Pretrained weights for finetuning (train with NaQ)

* [InternVideo](https://pan.baidu.com/s/1veKtGHzlKf-5kpEPiyNtsQ?pwd=id7a)
* [EgoVLP](https://pan.baidu.com/s/1pj2BzBC7hs-Lv31iP1Lk7A?pwd=wmcc)

### Ego4D-NLQ

| Feature     | Setting   |  NLQ v1                                                          | NLQ v2                                                          |
| ----------- | -----------|---------------------------------------------------- | --------------------------------------------------------------- |
| InternVideo | Finetuned| | [144](https://pan.baidu.com/s/15MeSuTg1I5sBMH-YKumbsw?pwd=a9fc) |
|EgoVLP| Finetuned|[173](https://pan.baidu.com/s/1rd49TkEw7ZOhNQe7Y0LD0g?pwd=bqp2) | |


### GoalStep

|Feature     | Setting   | GoalStep                                                        |
| ------------|------------|--------------------------------------- |
| InternVideo | Finetuned| [135](https://pan.baidu.com/s/1wT17nzokUk0FL-RFk_gFIA?pwd=jg7y) |

### TACoS

|Feature     | Setting   | Checkpoint                                                      |
|  --------- |--------- | --------------------------------------------------------------- |
|C3D        | Scratch   | [150](https://pan.baidu.com/s/1Lf-mSB8f8rDE_rtCZk4uDQ?pwd=a59n) |
|InternVideo| Finetuned | [131](https://pan.baidu.com/s/1k8tZdlZvmKmCT2sjOT-8jQ?pwd=3cfh) |

---

## Dataset / Benchmark

## Pretrain: [NaQ](https://github.com/srama2512/NaQ)

### Text Feature

Download features from [this Baidu Netdisk link](https://pan.baidu.com/s/1QXX-LMhUDSoky2Czh18bqA?pwd=tba6)

* narration feature: `narration_clip_token_features`
* narration jsonl: `format_unique_pretrain_data_v2.jsonl`

### Video Feature

The features are the same as the NLQ below.

* internvideo: `em_egovlp+internvideo_visual_features_1.87fps`

### Config

* 4 cards, total batch size is 16
* `configs/Ego4D-NLQ/v2/ego4d_nlq_v2_multitask_pretrain_2e-4.yaml`

---

## Ego4D-NLQ

### Feature Download

* Video Feature & Text Feature: GroundNLQ leverages the extracted egocentric InterVideo and EgoVLP features and CLIP textual token features. Please refer to [GroundNLQ](https://github.com/houzhijian/GroundNLQ).
* Download from [huggingface](https://huggingface.co/datasets/iLearn-Lab/CVPR25-OSGNet-Ego4D-NLQ)
* Download from baidu netdisk:

  * [Lavila Caption](https://pan.baidu.com/s/1ZeIOgf292gZwKKvYdZKqPA?pwd=p7qr)
  * Object Feature ([anno](https://pan.baidu.com/s/1WaRcaaWCSUZ_pFuKCRDC0w?pwd=kr9u), [classname](https://pan.baidu.com/s/186-WJ-mlRybTH8dmlrKzmA?pwd=5k2i))
  * Video Feature ([egovlp](https://pan.baidu.com/s/1cj3Egv3v4mVOi2nhfFrgkQ?pwd=4qnb))
  * Text Feature ([NLQ v1 feature](https://pan.baidu.com/s/1VTdgU2K_rxy0WWHMT_6MDw?pwd=h7gb))

### Text Feature

* NLQ v1 feature: `nlq_v1_clip_token_features`
* NLQ v2 feature: `nlq_v2_clip_token_features`
* egovideo: `egovideo_token_lmdb`

### Video Feature

* egovlp: `egovlp_lmdb`
* internvideo: `em_egovlp+internvideo_visual_features_1.87fps`
* egovideo: `egovideo_all_lmdb`

### Lavila Caption

* `lavila.zip`

### Object Feature

* anno: `co-detr/class-score0.6-minnum10-lmdb`
* classname: `classname-clip-base/a_photo_of.pt`

### Config

* 2 cards, total batch size is 8

**InternVideo**

* v1: `ego4d_nlq_v1_multitask_egovlp_256_finetune_2e-4.yaml`
* v2: `ego4d_nlq_v2_multitask_finetune_2e-4.yaml`

**EgoVideo**

* v2: `ego4d_nlq_v2_egovideo_finetune_4e-4.yaml`

---

## GoalStep

### Feature Download
* Download from [huggingface](https://huggingface.co/datasets/iLearn-Lab/CVPR25-OSGNet-Ego4D-GoalStep)
* Download from baidu netdisk:
  * [Text Feature](https://pan.baidu.com/s/1CwZhtSA3fzXA2brYcMjWCg?pwd=6991)
  * Video Feature ([clip](https://pan.baidu.com/s/1HOywlNFjeaGVWDnCdaLxUg?pwd=iig5), [not clip](https://pan.baidu.com/s/1Gna38KmKZdGl1uqOaEGGlw?pwd=iig5))
  * [lavila caption](https://pan.baidu.com/s/1syYuZf7H62TEnjkbw-uEmA?pwd=42hk)
  * Object Feature ([clip](https://pan.baidu.com/s/1ZnReOQhQ5-Zw0W1pJDhkHQ?pwd=xd5s), [not clip](https://pan.baidu.com/s/19BlDvo3AY1IfeO2DDmb65g?pwd=x4ha))

### Text Feature

* `clip_query_lmdb`

### Video Feature

* internvideo: `internvideo_clip_lmdb` (Due to memory limitations, we truncated the videos in the training set.), `internvideo_lmdb`

### Lavila Caption

* `lavila.zip`

### Object Feature

* anno: `co-detr/clip-class-lmdb` (after clip)
* classname: `classname-clip-base/a_photo_of.pt` (the same as Ego4D-NLQ)

### Config

* 4 cards, total batch size is 4
* finetune: `ego4d_goalstep_v2_baseline_2e-4.yaml`

---

## TACoS

### Feature Download
* Download from [huggingface](https://huggingface.co/datasets/iLearn-Lab/CVPR25-OSGNet-TACoS)
* Download features from [this Baidu Netdisk link](https://pan.baidu.com/s/1Zemfogt30ACGuOAZsmvx1A?pwd=arrt).

### Text Feature

* clip: `all_clip_token_features`
* glove: `glove_clip_token_features`

### Video Feature

* c3d: `c3d_lmdb`
* internvideo: `internvideo_lmdb`

### Lavila Caption

* `lavila.zip`

### Object Feature

* anno: `co-detr/class-score0.6-minnum10-lmdb`
* classname: `classname-clip-base/a_photo_of.pt` (the same as Ego4D-NLQ)

### Config

* 4 cards, total batch size is 8
* finetune: `tacos_baseline_1e-4.yaml`
* scratch: `tacos_c3d_glove_weight1_5e-5.yaml`

---

## Usage

We adopt distributed data parallel [DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) and fault-tolerant distributed training with [torchrun](https://pytorch.org/tutorials/beginner/ddp_series_fault_tolerance.html).

### Training from scratch

Training and pretraining can be launched by running:

```bash
bash tools/train.sh CONFIG_FILE False OUTPUT_PATH CUDA_DEVICE_ID MODE
```

where:

* `CONFIG_FILE` is the config file for model / dataset hyperparameter initialization
* `OUTPUT_PATH` is the model output directory name defined by yourself
* `CUDA_DEVICE_ID` is the CUDA device id
* `MODE` is the running mode

The checkpoints and other experiment log files will be written into `<output_folder>/OUTPUT_PATH`, where `output_folder` is defined in the config file.

#### Example: TACoS

```bash
bash tools/train.sh /home/feng_yi_sen/OSGNet/configs/tacos/tacos_c3d_glove_weight1_5e-5.yaml False objectmambafinetune219 0,1,2,3 train
```

### Finetuning

Training can be launched by running:

```bash
bash tools/train.sh CONFIG_FILE RESUME_PATH OUTPUT_PATH CUDA_DEVICE_ID MODE
```

where `RESUME_PATH` is the path to the pretrained model weights.

#### Example: Ego4D-NLQ v2

```bash
bash tools/train.sh configs/Ego4D-NLQ/v2/ego4d_nlq_v2_multitask_finetune_2e-4.yaml /root/autodl-tmp/model/GroundNLQ/ckpt/save/model_7_pretrain.pth.tar objectmambafinetune219 0,1 train
```

#### Example: GoalStep

For GoalStep, `MODE` should be `not-eval-loss`.

```bash
bash tools/train.sh configs/goalstep/ego4d_goalstep_v2_baseline_2e-4.yaml /root/autodl-tmp/model/GroundNLQ/ckpt/save/model_7_pretrain.pth.tar objectmambafinetune219 0,1 not-eval-loss
```

### Inference

Once the model is trained, you can use the following command for inference:

```bash
python eval_nlq.py CONFIG_FILE CHECKPOINT_PATH -gpu CUDA_DEVICE_ID
```

where `CHECKPOINT_PATH` is the path to the saved checkpoint.

#### Example: Ego4D-NLQ v2

```bash
python eval_nlq.py configs/Ego4D-NLQ/v2/ego4d_nlq_v2_multitask_finetune_2e-4.yaml /root/autodl-tmp/model/GroundNLQ/ckpt/ego4d_nlq_v2_multitask_finetune_2e-4_objectmambafinetune144/model_2_26.834358523725836.pth.tar -gpu 1
```

---

## Citation

If you are using our code, please consider citing our paper.

```bibtex
@inproceedings{feng2025object,
  title={Object-shot enhanced grounding network for egocentric video},
  author={Feng, Yisen and Zhang, Haoyu and Liu, Meng and Guan, Weili and Nie, Liqiang},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={24190--24200},
  year={2025}
}
```

```bibtex
@article{feng2025osgnet,
  title={OSGNet@ Ego4D Episodic Memory Challenge 2025},
  author={Feng, Yisen and Zhang, Haoyu and Chu, Qiaohui and Liu, Meng and Guan, Weili and Wang, Yaowei and Nie, Liqiang},
  journal={arXiv preprint arXiv:2506.03710},
  year={2025}
}
```

---

## Acknowledgement

This code is inspired by [GroundNLQ](https://github.com/houzhijian/GroundNLQ).

We use the same video and text features as GroundNLQ. We thank the authors for their awesome open-source contributions.

---

## License

This project is released under the MIT License. See [LICENSE](./LICENSE) for details.
