---
# Front matter. This is where you specify a lot of page variables.
layout: default
title:  "DEFORM"
date:   2023-06-16 03:03:01 -0400
description: >- # Supports markdown
  **D**ifferentiable Discrete **E**lastic Rods **F**or Deformable Linear **O**bjects with **R**eal-time **M**odeling
show-description: true

# Add page-specific mathjax functionality. Manage global setting in _config.yml
mathjax: true
# Automatically add permalinks to all headings
# https://github.com/allejo/jekyll-anchor-headings
autoanchor: false

# Preview image for social media cards
image:
  path: https://raw.githubusercontent.com/yich7045/DEFORM/blob/main/web_elements/DEFORM_model.jpg
  height: 100
  width: 256
  alt: Random Landscape
  
authors:
  - name: Yizhou Chen
    email: yizhouch@umich.edu
  - name: Yiting Zhang
    email: yitzhang@umich.edu 
  - name: Zachary Brei
    email: breizach@umich.edu
  - name: Tiancheng Zhang
    email: zhangtc@umich.edu
  - name: Yuzhen Chen
    email: yuzhench@umich.edu
  - name: Julie Wu
    email: jwuxx@umich.edu
  - name: Ram Vasudevan
    email: ramv@umich.edu

author-footnotes:
  All authors affiliated with the department of Mechanical Engineering and Department of Robotics of the University of Michigan, Ann Arbor.
  
  
links:
  - icon: arxiv
    icon-library: simpleicons
    text: Arxiv
    url: https://arxiv.org/abs/2406.05931
  - icon: github
    icon-library: simpleicons
    text: Code
    url: https://github.com/roahmlab/DEFORM
    

# End Front Matter
---

{% include sections/authors %}
{% include sections/links %}

---

# Abstract

This paper addresses the task of modeling Deformable Linear Objects (DLOs), such as ropes and cables, during dynamic motion over long time horizons.
This task presents significant challenges due to the complex dynamics of DLOs.
To address these challenges, this paper proposes differentiable Discrete Elastic Rods For deformable linear Objects with Real-time Modeling (DEFORM), 
a novel framework that combines a differentiable physics-based model with a learning framework to model DLOs accurately and in real-time. 
The performance of DEFORM is evaluated in an experimental setup involving two industrial robots and a variety of sensors.
A comprehensive series of experiments demonstrate the efficacy of DEFORM in terms of accuracy, computational speed, and generalizability when compared to state-of-the-art alternatives.
To further demonstrate the utility of DEFORM, this paper integrates it into a perception pipeline and illustrates its superior performance when compared to the state-of-the-art methods while tracking a DLO even in the presence of occlusions. 
Finally, this paper illustrates the superior performance of DEFORM when compared to state-of-the-art methods when it is applied to perform autonomous planning and control of DLOs.
<p align="center">
  <img src="https://raw.githubusercontent.com/yich7045/DEFT/main/web_elements/demo_image.png" class="img-responsive" alt="DEFORM model" style="width: 100%; height: auto;">

</p>
The figure shows DEFORM's predicted states (yellow) and the actual states (red) for a DLO over 4.5 seconds at 100 Hz. Note that the prediction is performed recursively, without requiring access to ground truth or perception during the process.

---


# Method
<div markdown="1" class="content-block grey justify no-pre">
DEFORM introduces a novel differentiable simulator as a physics prior for physics-informed learning to model DLOs in the real world. 
The following figure demonstrates the overview of DEFORM. Contributions of DEFORM are highlighted in green. 
a) DER models discretize DLOs into vertices, segment them into elastic rods, and model their dynamic propagation. 
DEFORM reformulates Discrete Elastic Rods(DER) into Differentiable DER (DDER) which describes how to compute gradients from the prediction loss, enabling efficient system identification and incorporation into deep learning pipelines.
b) To compensate for the error from DER's numerical integration, DEFORM introduces residual learning via DNNs.
c) 1 &rarr; 2: DER enforces inextensibility, but this does not satisfy classical conservation principles.  1 &rarr; 3: DEFORM enforces inextensibility with momentum conservation, which allows dynamic modeling while maintaining simulation stability.
<p align="center">
  <img src="https://raw.githubusercontent.com/yich7045/DEFORM/main/web_elements/DEFORM_Overview.png" class="img-responsive" alt="DEFORM overview" style="width: 100%; height: auto;">
</p>
</div>

---

# Dataset
<div markdown="1" class="content-block grey justify no-pre">
For each DLO, we collect 350 seconds of dynamic trajectory data in the real-world using the motion capture system at a frequency of 100 Hz. For dataset usage, please refer to train_DEFORM.py in [here](https://github.com/roahmlab/DEFORM).
</div>

---

# Demo Video
<div class="fullwidth">
<video controls="" width="100%">
    <source src="https://raw.githubusercontent.com/yich7045/DEFORM/main/web_elements/DEFORM_Demo.mp4" type="video/mp4">
</video>
</div>

<div markdown="1" class="content-block grey justify">

# [Citation](#citation)

This project was developed in [Robotics and Optimization for Analysis of Human Motion (ROAHM) Lab](http://www.roahmlab.com/) at University of Michigan - Ann Arbor.

```bibtex
@misc{chen2024differentiable,
      title={Differentiable Discrete Elastic Rods for Real-Time Modeling of Deformable Linear Objects}, 
      author={Yizhou Chen and Yiting Zhang and Zachary Brei and Tiancheng Zhang and Yuzhen Chen and Julie Wu and Ram Vasudevan},
      year={2024},
      eprint={2406.05931},
      archivePrefix={arXiv},
      primaryClass={id='cs.RO' full_name='Robotics' is_active=True alt_name=None in_archive='cs' is_general=False description='Roughly includes material in ACM Subject Class I.2.9.'}
}
```
</div>

---
