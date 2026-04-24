# Trigger Modulation via Motion Awareness and Sparse Keyframe Injection for Video Backdoor Attacks

This repository contains code for backdoor attack generation and evaluation on video facial-expression models:
- `Former-DFER`
- `M3DFEL`
- `PTH-Net`

## 1. Environment

```bash
conda create -n moska python=3.8
conda activate moska
pip install -r requirements.txt
```

## 2. Datasets

Download datasets from official sources and prepare frame folders/annotations used by this repo:
- [DFEW](https://dfew-dataset.github.io)
- [MAFW](https://github.com/MAFW-database/MAFW)
- [FERv39k](https://github.com/wangyanckxx/FERV39k)

`annotation/` already includes split files used by training scripts.

## 3. Generate Poisoned Data

### 3.1 Build temporal reference (`Face_avg_warped`)

```bash
python Backdoor_Attack/Motion_Compensation.py \
  --dataset FERv39k \
  --path /path/to/FERv39k/Frame \
  --save-path Face_avg_warped \
  --num-workers 4
```

### 3.2 Inject triggers (MATM)

```bash
python Backdoor_Attack/MATM.py \
  --dataset FERv39k \
  --root /path/to/FERv39k/Frame \
  --trigger-path /path/to/hello_kitty.png \
  --trigger-name hello_kitty \
  --poison-type WaNet \
  --ratio 0.1 \
  --use-avg-face \
  --use-flow
```

## 4. Train Attack Target Models

### 4.1 Former-DFER

```bash
python main_Former_DFER.py --is_temporal 5
```

### 4.2 M3DFEL

```bash
python main_M3DFEL.py \
  --dataset DFEW \
  --folds 1,2,3,4,5 \
  --poisons Poison_FFT \
  --is-temporal 5 \
  --is-key-frame true \
  --gpu-ids 0
```

### 4.3 PTH-Net

PTH-Net uses pre-extracted VideoMAE features (follow the official [PTH-Net](https://github.com/lm495455/PTH-Net) preprocessing first).

```bash
python main_PTH_Net.py \
  --dataset FERv39k \
  --folds 1 \
  --attack Poison_hello_kitty_avg_face_flow_0.1_0.2 \
  --data-root /path/to/features \
  --output-root ./outputs/pth_net \
  --gpu 0
```

## 5. Output Structure

Training scripts write logs/checkpoints/curves into per-experiment folders under `./outputs` by default.
