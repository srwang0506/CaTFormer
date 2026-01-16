<div align="center">

<h1 style="font-size: 2.2rem; border-bottom: none;">
  CaTFormer: Causal Temporal Transformer with Dynamic Contextual Fusion for Driving Intention Prediction
</h1>

<div style="font-size: 1.2rem; margin-bottom: 5px;">
  <a href="https://github.com/srwang0506" style="text-decoration: none;"><strong>Sirui Wang</strong></a><sup>†</sup>&emsp;
  <a href="https://github.com/Jokerealm" style="text-decoration: none;"><strong>Zhou Guan</strong></a><sup>†</sup>&emsp;
  <a href="https://github.com/pancacake" style="text-decoration: none;"><strong>Bingxi Zhao</strong></a>&emsp;
  <a href="https://github.com/Dean1217" style="text-decoration: none;"><strong>Tongjia Gu</strong></a>&emsp;
  <strong>Jie Liu</strong><sup>*</sup>
</div>

<div style="font-size: 1.1rem; font-weight: bold; font-style: italic; margin-bottom: 15px;">
  Beijing Jiaotong University
</div>

<div style="font-size: 0.9rem; color: #555; margin-bottom: 20px;">
  <sup>†</sup> Equal Contribution &emsp; <sup>*</sup> Corresponding Author
</div>

<div style="margin-bottom: 30px;">
  <a href="https://arxiv.org/abs/2507.13425">
    <img src="https://img.shields.io/badge/arXiv-2507.13425-b31b1b.svg?style=flat-square" alt="arXiv">
  </a>
  &emsp;
  <a href="https://github.com/srwang0506/CaTFormer">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717.svg?style=flat-square&logo=github" alt="GitHub">
  </a>
</div>

<img src="https://raw.githubusercontent.com/srwang0506/CaTFormer/main/pipeline.jpg" width="98%" alt="CaTFormer Pipeline" style="border-radius: 12px; box-shadow: 0 15px 30px rgba(0,0,0,0.1), 0 5px 15px rgba(0,0,0,0.05); border: 1px solid rgba(0,0,0,0.05);">
<br>

</div>

<br>

## 📰 News
* **[2026.01.09]** The preprint version is available on [arXiv](https://arxiv.org/abs/2507.13425).
* **[2025.11.08]** 🎉 Our paper has been accepted to **AAAI 2026**!

<br>

## 📝 Abstract

Accurate prediction of driving intention is key to enhancing the safety and interactive efficiency of human-machine co-driving systems. It serves as a cornerstone for achieving high-level autonomous driving. However, current approaches remain inadequate for accurately modeling the complex spatiotemporal interdependencies and the unpredictable variability of human driving behavior. To address these challenges, we propose **CaTFormer**, a causal Temporal Transformer that explicitly models causal interactions between driver behavior and environmental context for robust intention prediction. Specifically, CaTFormer introduces a novel **Reciprocal Delayed Fusion (RDF)** mechanism for precise temporal alignment of interior and exterior feature streams, a **Counterfactual Residual Encoding (CRE)** module that systematically eliminates spurious correlations to reveal authentic causal dependencies, and an innovative **Feature Synthesis Network (FSN)** that adaptively synthesizes these purified representations into coherent temporal representations. Experimental results demonstrate that CaTFormer attains state-of-the-art performance on the Brain4Cars dataset. It effectively captures complex causal temporal dependencies and enhances both the accuracy and transparency of driving intention prediction.

<br>

## 🚀 Getting Started

### Prerequisites

- Python >= 3.7
- PyTorch >= 1.7
- CUDA (for GPU support)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/srwang0506/CaTFormer.git
cd CaTFormer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dataset Preparation (Brain4Cars)

1) Download the [Brain4Cars dataset](https://www.brain4cars.com/) and extract all videos into JPG frame sequences (both interior and exterior cameras).

2) Directory layout we expect after preprocessing:
```
CaTFormer/
├── brain4cars_data/
│   ├── face_camera/
│   │   ├── end_action/
│   │   ├── lchange/
│   │   ├── lturn/
│   │   ├── rchange/
│   │   ├── rturn/
│   └── road_camera/
│       ├── end_action/
│       ├── lchange/
│       ├── lturn/
│       ├── rchange/
│       ├── rturn/
├── datasets/
│   └── annotation/      # fold0.csv, fold1.csv, ...
└── ...
```

3) Clone the official RAFT repo ([princeton-vl/RAFT](https://github.com/princeton-vl/RAFT)) and install the required dependencies **as instructed in its README** (incl. pretrained weights), then run `demo_brain4cars.py` to compute exterior optical flow; set the output path (or move results) so the processed flow frames are located at `brain4cars_data/road_camera/flow`.
```bash
git clone https://github.com/princeton-vl/RAFT.git
# follow RAFT README to set up env + download pretrained weights
python demo_brain4cars.py
# output should be placed under:
# brain4cars_data/road_camera/flow
```

After optical flow processing, the dataset directory structure is as follows:
```
CaTFormer/
├── brain4cars_data/
│   ├── face_camera/
│   │   ├── end_action/
│   │   ├── lchange/
│   │   ├── lturn/
│   │   ├── rchange/
│   │   ├── rturn/
│   └── road_camera/
│       └── flow/
│           ├── end_action/
│           ├── flow/
│               ├── end_action/
│               ├── lchange/
│               ├── lturn/
│               ├── rchange/
│               ├── rturn/
│           ├── lchange/
│           ├── lturn/
│           ├── rchange/
│           ├── rturn/
├── datasets/
│   └── annotation/      # fold0.csv, fold1.csv, ...
└── ...
```

4) Convert interior `face_camera` `.mat` metadata to `car_state.txt` before training/testing:
```bash
python extract_mat.py
# This writes `car_state.txt` beside each video folder for later loading.
```

<br>

## 🔧 Training

### Train on a Single Fold

To train the model on a specific fold (e.g., fold 3), use the provided shell script:
```bash
bash train_fold.sh
```

### Train on All Folds

To train on all folds for the 5-fold cross-validation:
```bash
bash train_total.sh
```

<br>

## 🧪 Testing

To evaluate the trained model, use the provided shell script:
```bash
bash test.sh
```

<br>

## 📚 Citation

If you find this work helpful, please consider citing:

```bibtex
@misc{wang2026catformercausaltemporaltransformer,
      title={CaTFormer: Causal Temporal Transformer with Dynamic Contextual Fusion for Driving Intention Prediction}, 
      author={Sirui Wang and Zhou Guan and Bingxi Zhao and Tongjia Gu and Jie Liu},
      year={2026},
      eprint={2507.13425},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.13425}, 
}
```

