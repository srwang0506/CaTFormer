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

<p align="justify">
Accurate prediction of driving intention is key to enhancing the safety and interactive efficiency of human-machine co-driving systems. It serves as a cornerstone for achieving high-level autonomous driving. However, current approaches remain inadequate for accurately modeling the complex spatiotemporal interdependencies and the unpredictable variability of human driving behavior. To address these challenges, we propose <strong>CaTFormer</strong>, a causal Temporal Transformer that explicitly models causal interactions between driver behavior and environmental context for robust intention prediction. Specifically, CaTFormer introduces a novel <strong>Reciprocal Delayed Fusion (RDF)</strong> mechanism for precise temporal alignment of interior and exterior feature streams, a <strong>Counterfactual Residual Encoding (CRE)</strong> module that systematically eliminates spurious correlations to reveal authentic causal dependencies, and an innovative <strong>Feature Synthesis Network (FSN)</strong> that adaptively synthesizes these purified representations into coherent temporal representations. Experimental results demonstrate that CaTFormer attains state-of-the-art performance on the Brain4Cars dataset. It effectively captures complex causal temporal dependencies and enhances both the accuracy and transparency of driving intention prediction.
</p>
