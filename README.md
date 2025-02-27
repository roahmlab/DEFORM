# Differentiable Discrete Elastic Rods for Real-Time Modeling of Deformable Linear Objects

This repository contains the source code for the paper [Differentiable Discrete Elastic Rods for Real-Time Modeling of Deformable Linear Objects](https://arxiv.org/abs/2406.05931). The paper has been accepted at CoRL 2024. Project page is [here](https://roahmlab.github.io/DEFORM/).

<p align="center">
  <img height="300" src="/demo_image.png"/>
</p>

This paper introduces DEFT, a novel framework that combines a differentiable physics-based model with a learning-based approach to accurately model and predict the dynamic behavior of Branched Deformable Linear Objects (BDLOs) in real time. As this paper illustrates, this model can be used in concert with a motion planning algorithm to autonomously manipulate BDLOs. The figures above illustrate how DEFT can be used to autonomously perform a wire insertion task.

**Left:** The system first plans a shape-matching motion, transitioning the BDLO from its initial configuration to the target shape (contoured with yellow), which serves as an intermediate waypoint.

**Right:** Starting from the intermediate configuration, the system performs thread insertion, guiding the BDLO into the target hole while also matching the target shape. Notably, DEFT predicts the shape of the wire recursively without relying on ground truth or perception data at any point in the process.

## Introduction
This paper addresses the task of modeling Deformable Linear Objects (DLOs), such as ropes and cables, during dynamic motion over long time horizons.This task presents significant challenges due to the complex dynamics of DLOs.To address these challenges, this paper proposes differentiable Discrete Elastic Rods For deformable linear Objects with Real-time Modeling (DEFORM), 
a novel framework that combines a differentiable physics-based model with a learning framework to model DLOs accurately and in real-time. 

**Authors:** Yizhou Chen (yizhouch@umich.edu), Yiting Zhang (yitzhang@umich.edu ), Zachary Brei (breizach@umich.edu), Tiancheng Zhang (zhangtc@umich.edu ), Yuzhen Chen (yuzhench@umich.edu), Julie Wu (jwuxx@umich.edu) and Ram Vasudevan (ramv@umich.edu).

All authors are affiliated with the Robotics department and the department of Mechanical Engineering of the University of Michigan, 2505 Hayward Street, Ann Arbor, Michigan, USA.

## Dependency 
- Run `pip install -r requirements.txt` to collect all python dependencies.

## Training DEFORM Models
- Example: run `python train_DEFORM.py  --DLO_type="DLO1"` to train a DEFORM model with the DLO1 dataset.
- Note that running mode = "evaluation_numpy" only works for batch = 1 for now.

## Dataset
- For each DLO, we collect 350 seconds of dynamic trajectory data in the real-world using the motion capture system at a frequency of 100 Hz. For dataset usage, please refer to train_DEFORM.py


## Citation
If you use DEFORM in an academic work, please cite using the following BibTex entry:
```
@article{DEFORM,
      title={Differentiable Discrete Elastic Rods for Real-Time Modeling of Deformable Linear Objects}, 
      author={Yizhou Chen and Yiting Zhang and Zachary Brei and Tiancheng Zhang and Yuzhen Chen and Julie Wu and Ram Vasudevan},
      journal={https://arxiv.org/abs/2406.05931},
      year={2024}
}
```

