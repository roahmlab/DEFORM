# Differentiable Discrete Elastic Rods for Real-Time Modeling of Deformable Linear Objects

This repository contains the source code for the paper [Differentiable Discrete Elastic Rods for Real-Time Modeling of Deformable Linear Objects](https://arxiv.org/abs/2406.05931). The paper has been accepted at CoRL 2024.

<p align="center">
  <img height="300" src="/demo_image.png"/>
</p>

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

## Citation
If you use DEFORM in an academic work, please cite using the following BibTex entry:
```
@misc{chen2024differentiable,
      title={Differentiable Discrete Elastic Rods for Real-Time Modeling of Deformable Linear Objects}, 
      author={Yizhou Chen and Yiting Zhang and Zachary Brei and Tiancheng Zhang and Yuzhen Chen and Julie Wu and Ram Vasudevan},
      year={2024},
      eprint={2406.05931},
      archivePrefix={arXiv},
      primaryClass={id='cs.RO' full_name='Robotics' is_active=True alt_name=None in_archive='cs' is_general=False description='Roughly includes material in ACM Subject Class I.2.9.'}
}
```

