This repository contains the implementation of the following paper:

 666666666666666666加油

# Get Started

## Environment Installation

```bash
python -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dataset Preparation

1. To download DIV2K, Flickr2K and CelebA, please refer to https://github.com/andreas128/SRFlow

2. Please put the downloaded DIV2K, Flickr2K and CelebA dataset in  
    data/raw/DIV2K, 
    data/raw/Flickr2K, 
    data/raw/CelebA, 
    respectively

3. Pack to pickle for training
```bash
# pack the CelebA dataset
python data_gen/celeb_a.py --config configs/celeb_a.yaml 
# pack the DIV2K dataset
python data_gen/df2k.py --config configs/df2k4x.yaml
```

## Pretrained Model
Please go to https://github.com/LeiaLi/SRDiff/releases/tag/v1.0.0 to download the pretrained models.
- CelebA
    - srdiff_pretrained_celebA/model_ckpt_steps_300000.ckpt
- DIV2K
    - srdiff_pretrained_div2k/model_ckpt_steps_400000.ckpt

## Train & Evaluate
1. Prepare datasets. Please refer to Dataset Preparation.
2. Modify config files.
    - CelebA
        - rrdb: configs/rrdb/celeb_a_pretrain.yaml
        - srdiff: configs/celeb_a.yaml
    - DIV2K
        - rrdb: configs/rrdb/df2k4x_pretrain.yaml
        - srdiff: configs/diffsr_df2k4x.yaml
3. Run training / evaluation code. The code is for training on 1 GPU.

### CelebA

```bash
# train rrdb-based conditional net
CUDA_VISIBLE_DEVICES=0 python tasks/trainer.py --config configs/rrdb/celeb_a_pretrain.yaml --exp_name rrdb_celebA_1 --reset
# train srdiff
CUDA_VISIBLE_DEVICES=0 python tasks/trainer.py --config configs/diffsr_celeb.yaml --exp_name diffsr_celebA_1 --reset --hparams="rrdb_ckpt=checkpoints/rrdb_celebA_1"

# tensorboard
tensorboard --logdir checkpoints/diffsr_celebA_1

# evaluate
CUDA_VISIBLE_DEVICES=0 python tasks/trainer.py --config configs/diffsr_celeb.yaml --exp_name diffsr_celebA_1 --infer

# evaluate with pretrained model
CUDA_VISIBLE_DEVICES=0 python tasks/trainer.py --config configs/diffsr_celeb.yaml --exp_name srdiff_pretrained_celebA --infer
```

### DIV2K

```bash
# train rrdb-based conditional net
CUDA_VISIBLE_DEVICES=0 python tasks/trainer.py --config configs/rrdb/df2k4x_pretrain.yaml --exp_name rrdb_div2k_1 --reset
# train srdiff
CUDA_VISIBLE_DEVICES=0 python tasks/trainer.py --config configs/diffsr_df2k4x.yaml --exp_name diffsr_div2k_1 --reset --hparams="rrdb_ckpt=checkpoints/rrdb_div2k_1"

# tensorboard
tensorboard --logdir checkpoints/diffsr_div2k_1

# evaluate
CUDA_VISIBLE_DEVICES=0 python tasks/trainer.py --config configs/diffsr_df2k4x.yaml --exp_name diffsr_div2k_1 --infer

# evaluate with pretrained model
CUDA_VISIBLE_DEVICES=0 python tasks/trainer.py --config configs/diffsr_df2k4x.yaml --exp_name srdiff_pretrained_div2k --infer
```


