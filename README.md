# Object-Shot Enhanced Grounding Network for Egocentric Video (*CVPR 2025*)

[Techical report](https://arxiv.org/abs/2505.04270)



This repo supports data pre-processing, training and evaluation of the Ego4D-NLQ, Ego4D-GoalStep, TACoS dataset. 




## Install-dependencies 
* Follow [INSTALL.sh](./install/install.sh) for installing necessary dependencies and compiling the code.Torch version recommand >=1.8.0

### Prepare-offline-data
* Required Feature: text feature, video feature, lavila caption(need to unzip), object feature
## Ego4D-NLQ
### Feature Download
* Video Feature & Text Feature: GroundNLQ leverage the extracted egocentric InterVideo and EgoVLP features and CLIP textual token features, please refer to [GroundNLQ](https://github.com/houzhijian/GroundNLQ).
### Text Feature
* narration feature: narration_clip_token_features
* narration jsonl: format_unique_pretrain_data_v2.jsonl
* NLQ v2 feature: nlq_v2_clip_token_features
### Video Feature
* egovlp: egovlp_lmdb
* internvideo : em_egovlp+internvideo_visual_features_1.87fps
### Lavila Caption 
* lavila.zip
### Object Feature
* anno: co-detr/class-score0.6-lmdb
* classname: classname-clip-base/a_photo_of.pt
### Config
2 cards, total batch is 8.
* v1: ego4d_nlq_v1_multitask_egovlp_256_finetune_2e-4.yaml
* v2: ego4d_nlq_v2_multitask_finetune_2e-4.yaml
### checkpoints
| NLQ v1 s | NLQ v1 f | NLQ v2 s | NLQ v2 f |
|----------|----------|----------|----------|
|165       | 173      | 203      | 144      |

## GoalStep
### Text Feature
* clip_query_lmdb
### Video Feature
* internvideo: internvideo_clip_lmdb
### Lavila Caption 
* lavila.zip
### Object Feature
* anno after clip: co-detr/clip-class-lmdb
* classname: classname-clip-base/a_photo_of.pt(the same as Ego4D-NLQ)
### config
4 cards, total batch is 4.
* finetune: ego4d_goalstep_v2_baseline_2e-4.yaml
### checkpoints
| GoalStep |
|----------|
| 135      |
## TACoS
### Feature Download
* Download features from [this Baidu Netdisk link](https://pan.baidu.com/s/1Zemfogt30ACGuOAZsmvx1A?pwd=arrt).
### Text Feature
* clip: all_clip_token_features
* glove: glove_clip_token_features
### Video Feature
* c3d: c3d_lmdb
* internvideo: internvideo_lmdb
### Lavila Caption 
* lavila.zip
### Object Feature
* anno: co-detr/class-score0.6-minnum10-lmdb
### config
4 cards, total batch is 8.
* finetune: tacos_baseline_1e-4.yaml
* scratch: tacos_c3d_glove_weight1_5e-5.yaml 
### checkpoints
| Setting | Checkpoint |
|---------|------------|
| Scratch |[150](https://pan.baidu.com/s/1Lf-mSB8f8rDE_rtCZk4uDQ?pwd=a59n)|
|Finetuned|[131](https://pan.baidu.com/s/1k8tZdlZvmKmCT2sjOT-8jQ?pwd=3cfh)|
## Code-Overview

* ./libs/core: Parameter default configuration module.
  * ./configs: Parameter file.
* ./ego4d_data: the annotation data.
* ./tools: Scripts for running.
* ./libs/datasets: Data loader and IO module.
* ./libs/modeling: Our main model with all its building blocks.
* ./libs/utils: Utility functions for training, inference, and postprocessing.

##  Experiments
We adopt distributed data parallel [DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) and 
fault-tolerant distributed training with [torchrun](https://pytorch.org/tutorials/beginner/ddp_series_fault_tolerance.html).


### Training-From-Scratch
Training and pretraining can be launched by running the following command:
```
bash tools/train.sh CONFIG_FILE False OUTPUT_PATH CUDA_DEVICE_ID MODE
```
where `CONFIG_FILE` is the config file for model/dataset hyperparameter initialization,
`EXP_ID` is the model output directory name defined by yourself, `CUDA_DEVICE_ID` is cuda device id.
The checkpoints and other experiment log files will be written into `<output_folder>/OUTPUT_PATH`, output_folder is defined in the config file. 

Take TACoS as example:
```
bash tools/train.sh /home/feng_yi_sen/OSGNet/configs/tacos/tacos_c3d_glove_weight1_5e-5.yaml False objectmambafinetune219 0,1,2,3 train
```

### Training-Finetune
Training can be launched by running the following command:
```
bash tools/train.sh CONFIG_FILE RESUME_PATH OUTPUT_PATH CUDA_DEVICE_ID MODE
```
where `RESUME_PATH` is the path of the pretrained model weights.

The config file is the same as scratch.

Take Ego4D-NLQ v2 as example:
```
bash tools/train.sh configs/Ego4D-NLQ/v2/ego4d_nlq_v2_multitask_finetune_2e-4.yaml /root/autodl-tmp/model/GroundNLQ/ckpt/save/model_7_pretrain.pth.tar objectmambafinetune219 0,1 train
```
For GoalStep, mode should be not-eval-loss.
```
bash tools/train.sh configs/goalstep/ego4d_goalstep_v2_baseline_2e-4.yaml /root/autodl-tmp/model/GroundNLQ/ckpt/save/model_7_pretrain.pth.tar objectmambafinetune219 0,1 not-eval-loss
```
### Inference
Once the model is trained, you can use the following commands for inference:
```
python eval_nlq.py CONFIG_FILE CHECKPOINT_PATH -gpu CUDA_DEVICE_ID 
```
where `CHECKPOINT_PATH` is the path to the saved checkpoint,`save` is for controling the output . 
Take Ego4D-NLQ v2 as example:
```
python eval_nlq.py configs/Ego4D-NLQ/v2/ego4d_nlq_v2_multitask_finetune_2e-4.yaml  /root/autodl-tmp/model/GroundNLQ/ckpt/ego4d_nlq_v2_multitask_finetune_2e-4_objectmambafinetune144/model_2_26.834358523725836.pth.tar -gpu 1
```

## Citation
If you are using our code, please consider citing our paper.
```
@article{feng2025object,
  title={Object-Shot Enhanced Grounding Network for Egocentric Video},
  author={Feng, Yisen and Zhang, Haoyu and Liu, Meng and Guan, Weili and Nie, Liqiang},
  journal={arXiv preprint arXiv:2505.04270},
  year={2025}
}
```
## Acknowledgements
This code is inspired by [GroundNLQ](https://github.com/houzhijian/GroundNLQ). 
We use the same video and text feature as GroundNLQ. 
We thank the authors for their awesome open-source contributions. 