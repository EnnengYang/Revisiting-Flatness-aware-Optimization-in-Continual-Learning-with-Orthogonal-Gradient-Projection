# Revisiting-Flatness-aware-Optimization-in-Continual-Learning-with-Orthogonal-Gradient-Projection

A repository of **'[Revisiting Flatness-aware Optimization in Continual Learning with Orthogonal Gradient Projection. TPAMI, 2025.](https://ieeexplore.ieee.org/abstract/document/10874188)'**. This paper is an extended journal version of **'[Data Augmented Flatness-aware Gradient Projection for Continual Learning. ICCV, 2023.](https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_Data_Augmented_Flatness-aware_Gradient_Projection_for_Continual_Learning_ICCV_2023_paper.pdf)'**.


## Abstract
> The goal of continual learning (CL) is to learn from a series of continuously arriving new tasks without forgetting previously learned old tasks. To avoid catastrophic forgetting of old tasks, orthogonal gradient projection (OGP) based CL methods constrain the gradients of new tasks to be orthogonal to the space spanned by old tasks. This strict gradient constraint will limit the learning ability of new tasks, resulting in lower performance on new tasks. In this paper, we first establish a unified framework for OGP-based CL methods. We then revisit OGP-based CL methods from a new perspective on the loss landscape, where we find that when relaxing projection constraints to improve performance on new tasks, the unflatness of the loss landscape can lead to catastrophic forgetting of old tasks. Based on our findings, we propose a new Dual Flatness-aware OGD framework that optimizes the flatness of the loss landscape from both data and weight levels. Our framework consists of three modules: data and weight perturbation, flatness-aware optimization, and gradient projection. Specifically, we first perform perturbations on the task's data and current model weights to make the task's loss reach the worst-case. Next, we optimize the loss and loss landscape on the original data and the worst-case perturbed data to obtain a flatness-aware gradient. Finally, the flatness-aware gradient will update the network in directions orthogonal to the space spanned by the old tasks. Extensive experiments on four benchmark datasets show that the framework improves the flatness of the loss landscape and performance on new tasks, and achieves state-of-the-art (SOTA) performance on average accuracy across all tasks.

## Citation
If you find our paper or this resource helpful, please consider cite:
```
@article{yang2025revisiting_TPAMI_2025,
  title={Revisiting Flatness-aware Optimization in Continual Learning with Orthogonal Gradient Projection},
  author={Yang, Enneng and Shen, Li and Wang, Zhenyi and Liu, Shiwei and Guo, Guibing and Wang, Xingwei and Tao, Dacheng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025},
  publisher={IEEE}
}

@InProceedings{DFGP_ICCV_2023,
    author    = {Yang, Enneng and Shen, Li and Wang, Zhenyi and Liu, Shiwei and Guo, Guibing and Wang, Xingwei},
    title     = {Data Augmented Flatness-aware Gradient Projection for Continual Learning},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2023},
    pages     = {5630-5639}
}
```
Thanks!

## Code
- Please refer to the  '[data](https://github.com/EnnengYang/Revisiting-Flatness-aware-Optimization-in-Continual-Learning-with-Orthogonal-Gradient-Projection/tree/main/data)' directory to configure the experiment datasets.
- Please refer to the '[code](https://github.com/EnnengYang/Revisiting-Flatness-aware-Optimization-in-Continual-Learning-with-Orthogonal-Gradient-Projection/tree/main/code)' directory to run the main experiment code.
- Please refer to the '[log](https://github.com/EnnengYang/Revisiting-Flatness-aware-Optimization-in-Continual-Learning-with-Orthogonal-Gradient-Projection/tree/main/log)' directory for major hyperparameter configurations and experimental results.

## Acknowledgement
Our implementation mainly  references the code below, thanks to them.
- https://github.com/EnnengYang/DFGP
- https://github.com/sahagobinda/GPM
- https://github.com/LYang-666/TRGP
- https://github.com/sahagobinda/SGP
- https://github.com/THUNLP-MT/ROGO
- https://github.com/davda54/sam

Thanks to other more GitHub open source code.
