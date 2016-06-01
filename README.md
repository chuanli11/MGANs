# MGANs
Training & Testing code (torch), pre-trained models and supplementary materials for "[Precomputed Real-Time Texture Synthesis with Markovian Generative Adversarial Networks](http://arxiv.org/abs/1604.04382)". 

See this [video](https://www.youtube.com/watch?v=PRD8LpPvdHI) for a quick explaination for our method and results. 

# Setup
This code is based on Torch. It has only been tested on Mac and Ubuntu.

Dependencies:
* [Torch](https://github.com/torch/torch7)

For CUDA backend:
* [CUDA](https://developer.nvidia.com/cuda-downloads)
* [cudnn](https://developer.nvidia.com/cudnn)


# Training
Simply cd into folder "code/" and run the training script.

```bash
th train.lua
```

The current script is an example of training a network from 25 ImageNet photos and a painting from Van Gogh ("Olive Trees with the Alpilles in the Background"). The input data are organized in the following way: 
  * "Dataset/VG_Alpilles_ImageNet100/ContentInitial": 5 training ImageNet photos to initialize the discriminator.
  * "Dataset/VG_Alpilles_ImageNet100/ContentTrain": 100 training ImageNet photos.
  * "Dataset/VG_Alpilles_ImageNet100/ContentTest": 10 testing ImageNet photos (for later inspection).
  * "Dataset/VG_Alpilles_ImageNet100/Style": Van Gogh's painting.

The training process has three main steps: 
  * Use MDAN to generate training images (MDAN_wrapper.lua). 
  * Data Augmentation (AG_wrapper.lua).
  * Train MGAN (MDAN_wrapper.lua).

# Testing
The testing process has two steps:
* Step 1: call "th release_MGAN.lua" to concatenate the VGG encoder with the generator. 
* Step 2: call "th demo_MGAN.lua" to test the network with new photos.

# Acknowledgement
* We thank Soumith Chintala for sharing his implementation of [Deep Convolutional Generative Adversarial Networks](https://github.com/soumith/dcgan.torch). 
* We thank the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) team and the [ImageNet](http://image-net.org/) team for sharing their dataset. 
