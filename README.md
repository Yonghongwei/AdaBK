# AdaBK Algorithm
## [Paper](https://www4.comp.polyu.edu.hk/~cslzhang/paper/CVPR2023_AdaBK.pdf) | [Supplementary Material](https://www4.comp.polyu.edu.hk/~cslzhang/paper/CVPR2023_AdaBK_supp.pdf)
A General Regret Bound of Preconditioned Gradient Method for DNN Training

[Hongwei Yong](https://sites.google.com/view/yonghongwei-homepage), Ying Sun and [Lei Zhang](https://www4.comp.polyu.edu.hk/~cslzhang/)

## Abstract 
While adaptive learning rate methods, such as Adam, have achieved remarkable improvement in optimizing Deep Neural Networks (DNNs), they consider only the diagonal elements of the full preconditioned matrix. Though the full-matrix preconditioned gradient methods theoretically have a lower regret bound, they are impractical for use to train DNNs because of the high complexity. In this paper, we present a general regret bound with a constrained full-matrix preconditioned gradient, and show that the updating formula of the preconditioner can be derived by solving a cone-constrained optimization problem. With the block-diagonal and Kronecker-factorized constraints, a specific guide function can be obtained. By minimizing the upper bound of the guide function, we develop a new DNN optimizer, termed AdaBK. A series of techniques, including statistics updating, dampening, efficient matrix inverse root computation, and gradient amplitude preservation, are developed to make AdaBK effective and efficient to implement. The proposed AdaBK can be readily embedded into many existing DNN optimizers, e.g., SGDM and AdamW, and the corresponding SGDM\_BK and AdamW\_BK algorithms demonstrate significant improvements over existing DNN optimizers on benchmark vision tasks, including image classification, object detection and segmentation.

## A General Regret Bound for Constrained Preconditioned Gradient
<div  align="center"><img src="https://github.com/Yonghongwei/AdaBK/blob/main/image/1.png" height="75%" width="75%" alt="General Regret Bound Theorem"/></div>
## Proposed Algorithm
* AdaBK algorithm 
<div  align="center"><img src="https://github.com/Yonghongwei/AdaBK/blob/main/image/6.png" height="75%" width="75%" alt="EFW algorithm"/></div>

* SGDM\_BK and AdamW\_BK algorithm
<div  align="center"><img src="https://github.com/Yonghongwei/AdaBK/blob/main/image/4.png" height="75%" width="75%" alt="WSGDM and WAdam algorithm"/></div>

## Citation
    @inproceedings{Hongwei2023AdaBK,
      title={A General Regret Bound of Preconditioned Gradient Method for DNN Training},
      author={Hongwei Yong, Ying Sun and Lei Zhang},
      booktitle={IEEE conference on computer vision and pattern recognition},
      year={2023}
    }

