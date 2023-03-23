# W-SGDM-and-W-Adam-DNN-optimizers
## [Paper](http://www4.comp.polyu.edu.hk/~cslzhang/paper/ECCV2022-EFW.pdf) | [Supplementary Material](http://www4.comp.polyu.edu.hk/~cslzhang/paper/ECCV2022-EFW-supp.pdf)
An Embedded Feature Whitening Approach to Deep Neural Network Optimization

[Hongwei Yong](https://sites.google.com/view/yonghongwei-homepage) and [Lei Zhang](https://www4.comp.polyu.edu.hk/~cslzhang/)

## Abstract 
Compared with the feature normalization methods that are widely used in deep neural network (DNN) training, feature whitening methods take the correlation of features into consideration, which can help to learn more effective features. However, existing feature whitening methods have several limitations, such as the large computation and memory cost, inapplicable to pre-trained DNN models, the introduction of additional parameters, etc., making them impractical to use in optimizing DNNs. To overcome these drawbacks, we propose a novel Embedded Feature Whitening (EFW) approach to DNN optimization. EFW only adjusts the gradient of weight by using the whitening matrix without changing any part of the network so that it can be easily adopted to optimize pre-trained and well-defined DNN architectures. The momentum, adaptive dampening and gradient norm recovery techniques associated with EFW are consequently developed to make its implementation efficient with acceptable extra computation and memory cost. We apply EFW to two commonly used DNN optimizers, \ie, SGDM and Adam (or AdamW), and name the obtained optimizers as W-SGDM and W-Adam. Extensive experimental results on various vision tasks, including image classification, object detection, segmentation and person ReID, demonstrate the superiority of W-SGDM and W-Adam to state-of-the-art DNN optimizers.

## Proposed Algorithm
* EFW algorithm 
<div  align="center"><img src="https://github.com/Yonghongwei/W-SGDM-and-W-Adam/blob/main/EFW_algorithm.png" height="75%" width="75%" alt="EFW algorithm"/></div>

* WSGDM and WAdam algorithm
<div  align="center"><img src="https://github.com/Yonghongwei/W-SGDM-and-W-Adam/blob/main/WSGDM_WAdam.png" height="75%" width="75%" alt="WSGDM and WAdam algorithm"/></div>

## Citation
    @inproceedings{Hongwei2022EFW,
      title={An Embedded Feature Whitening Approach to Deep Neural Network Optimization},
      author={Hongwei Yong and Lei Zhang},
      booktitle={the European Conference on Conputer Vision},
      year={2022}
    }

