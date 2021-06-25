# FGVC8

[Exploring Vision Transformers for Fine-grained Classification](https://arxiv.org/abs/2106.10587) paper presented at the [CVPR 2021, The Eight Workshop on Fine-Grained Visual Categorization](https://sites.google.com/view/fgvc8) on June 25th.

### Abstract
Existing computer vision research in categorization struggles with fine-grained attributes recognition due to the inherently high intra-class variances and low inter-class variances. SOTA methods tackle this challenge by locating the most informative image regions and rely on them to classify the complete image. The most recent work, Vision Transformer (ViT), shows its strong performance in both traditional and fine-grained classification tasks.

In this work, we propose a **multi-stage** ViT framework for fine-grained image classification tasks, which localizes the informative image regions without requiring architectural changes using the inherent multi-head **self-attention** mechanism. We also introduce **attention-guided augmentations** for improving the model's capabilities.

We demonstrate the value of our approach by experimenting with four popular fine-grained benchmarks: CUB-200-2011, Stanford Cars, Stanford Dogs, and FGVC7 Plant Pathology. We also prove our model's interpretability via qualitative results.

<img src="https://i.ibb.co/M8wxmW0/poster.png" alt="poster" border="0">

### Instructions

Upcoming

### Citation

If you find interesting our results, or you use or code/ideas please consider to cite our work:

```
@misc{conde2021exploring,
      title={Exploring Vision Transformers for Fine-grained Classification}, 
      author={Marcos V. Conde and Kerem Turgutlu},
      year={2021},
      eprint={2106.10587},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

### References

- [Multi-branch and Multi-scale Attention Learning for Fine-Grained Visual Categorization](https://arxiv.org/pdf/2003.09150.pdf)
- [Look Closer to See Better: Recurrent Attention Convolutional Neural Network
for Fine-grained Image Recognition](https://openaccess.thecvf.com/content_cvpr_2017/papers/Fu_Look_Closer_to_CVPR_2017_paper.pdf)
- [See Better Before Looking Closer: Weakly Supervised Data Augmentation Network for Fine-Grained Visual Classification](https://arxiv.org/abs/1901.09891)

