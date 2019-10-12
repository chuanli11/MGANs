# MGANs
Training & Testing code (torch), pre-trained models and supplementary materials for "[Precomputed Real-Time Texture Synthesis with Markovian Generative Adversarial Networks](http://arxiv.org/abs/1604.04382)". 

See this [video](https://www.youtube.com/watch?v=PRD8LpPvdHI) for a quick explaination for our method and results. 

# Setup

As building Torch with the latest CUDA is a troublesome work, we recommend following the following steps to people who want to reproduce the results: 
It has been tested on Ubuntu with CUDA 10.

__Step One: Install CUDA 10 and CUDNN 7.6.2__

If you have a fresh Ubuntu, we recommend [Lambda Stack](https://lambdalabs.com/lambda-stack-deep-learning-software) which helps you install the latest drivers, libraries, and frameworks for deep learning. Otherwise, you can install the CUDA toolkit and CUDNN from these links:
* [CUDA](https://developer.nvidia.com/cuda-downloads)
* [CUDNN](https://developer.nvidia.com/cudnn)

__Step Two: Install Torch__
```
git clone https://github.com/nagadomi/distro.git ~/torch --recursive
cd ~/torch
./install-deps
./clean.sh
./update.sh

. ~/torch/install/bin/torch-activate
sudo apt-get install libprotobuf-dev protobuf-compiler
luarocks install loadcaffe
```

# Demo

```
cd code
th demo_MGAN.lua
```

# Training
Simply cd into folder "code/" and run the training script.

```bash
th train.lua
```

The current script is an example of training a network from 100 ImageNet photos and a single painting from Van Gogh. The input data are organized in the following way: 
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

## Display
You can use the browser based [display package](https://github.com/szym/display) to display the training process for both MDANs and MGANs.
  * Install: `luarocks install https://raw.githubusercontent.com/szym/display/master/display-scm-0.rockspec`
  * Call: `th -ldisplay.start`
  * See results at this URL: [http://localhost:8000](http://localhost:8000)

# Example
We chose Van Gogh's "Olive Trees with the Alpilles in the Background" as the reference texture.
<p><a href="/Dataset/VG_Alpilles_ImageNet100/Style/VG_Apilles.png" target="_blank"><img src="/Dataset/VG_Alpilles_ImageNet100/Style/VG_Apilles.png" height="220px" style="max-width:100%;"></a></p>

We then transfer 100 ImageNet photos into the same style with the proposed MDANs method. MDANs take an iterative deconvolutional approach, which is similar to "[A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576)" by Leon A. Gatys et al. and our previous work "[CNNMRF](https://github.com/chuanli11/CNNMRF)". Differently, it uses adversarial training instead of gaussian statistics ("[A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576)) or nearest neighbour search "[CNNMRF](https://github.com/chuanli11/CNNMRF)". Here are some transferred results from MDANs:
<p>
<a href="/pictures/ILSVRC2012_val_00000003.png" target="_blank"><img src="/pictures/ILSVRC2012_val_00000003.png" height="120px" style="max-width:100%;"></a>
<a href="/pictures/ILSVRC2012_val_00000034.png" target="_blank"><img src="/pictures/ILSVRC2012_val_00000034.png" height="120px" style="max-width:100%;"></a>
<a href="/pictures/ILSVRC2012_val_00000015.png" target="_blank"><img src="/pictures/ILSVRC2012_val_00000015.png" height="120px" style="max-width:100%;"></a>
<a href="/pictures/ILSVRC2012_val_00000032.png" target="_blank"><img src="/pictures/ILSVRC2012_val_00000032.png" height="120px" style="max-width:100%;"></a>
<a href="/pictures/ILSVRC2012_val_00000033.png" target="_blank"><img src="/pictures/ILSVRC2012_val_00000033.png" height="120px" style="max-width:100%;"></a>
<a href="/pictures/ILSVRC2012_val_00000003_MDANs.png" target="_blank"><img src="/pictures/ILSVRC2012_val_00000003_MDANs.png" height="120px" style="max-width:100%;"></a>
<a href="/pictures/ILSVRC2012_val_00000034_MDANs.png" target="_blank"><img src="/pictures/ILSVRC2012_val_00000034_MDANs.png" height="120px" style="max-width:100%;"></a>
<a href="/pictures/ILSVRC2012_val_00000015_MDANs.png" target="_blank"><img src="/pictures/ILSVRC2012_val_00000015_MDANs.png" height="120px" style="max-width:100%;"></a>
<a href="/pictures/ILSVRC2012_val_00000032_MDANs.png" target="_blank"><img src="/pictures/ILSVRC2012_val_00000032_MDANs.png" height="120px" style="max-width:100%;"></a>
<a href="/pictures/ILSVRC2012_val_00000033_MDANs.png" target="_blank"><img src="/pictures/ILSVRC2012_val_00000033_MDANs.png" height="120px" style="max-width:100%;"></a>
</p>

The results look nice, so we know adversarial training is able to produce results that are comparable to previous methods. In other experiments we observed that gaussian statistics work remarkable well for painterly textures, but can sometimes be too flexible for photorealistic textures; nearest-neighbor search preserve photorealistic details but can be too rigid for deformable textures. In some sense MDANs offers a relatively more balanced choice with advaserial training. See our paper for more discussoins.

Like previous deconvolutional methods, MDANs is VERY slow. A Nvidia Titan X takes about one minute to transfer a photo of 384 squared. To make it faster, we replace the deconvolutional process by a feed-forward network (MGANs). The feed-forward network takes long time to train (45 minutes for this example on a Titan X), but offers significant speed up in testing time. Here are some results from MGANs:

<p>
<a href="/pictures/ILSVRC2012_val_00000511.png" target="_blank"><img src="/pictures/ILSVRC2012_val_00000511.png" height="150px" style="max-width:100%;"></a>
<a href="/pictures/ILSVRC2012_val_00000522.png" target="_blank"><img src="/pictures/ILSVRC2012_val_00000522.png" height="150px" style="max-width:100%;"></a>
<a href="/pictures/ILSVRC2012_val_00000523.png" target="_blank"><img src="/pictures/ILSVRC2012_val_00000523.png" height="150px" style="max-width:100%;"></a>
<a href="/pictures/ILSVRC2012_val_00000534.png" target="_blank"><img src="/pictures/ILSVRC2012_val_00000534.png" height="150px" style="max-width:100%;"></a>
<a href="/pictures/ILSVRC2012_val_00000537.png" target="_blank"><img src="/pictures/ILSVRC2012_val_00000537.png" height="150px" style="max-width:100%;"></a>
<a href="/pictures/ILSVRC2012_val_00000511_MGANs.jpg" target="_blank"><img src="/pictures/ILSVRC2012_val_00000511_MGANs.jpg" height="148px" style="max-width:100%;"></a>
<a href="/pictures/ILSVRC2012_val_00000522_MGANs.jpg" target="_blank"><img src="/pictures/ILSVRC2012_val_00000522_MGANs.jpg" height="148px" style="max-width:100%;"></a>
<a href="/pictures/ILSVRC2012_val_00000523_MGANs.jpg" target="_blank"><img src="/pictures/ILSVRC2012_val_00000523_MGANs.jpg" height="148px" style="max-width:100%;"></a>
<a href="/pictures/ILSVRC2012_val_00000534_MGANs.jpg" target="_blank"><img src="/pictures/ILSVRC2012_val_00000534_MGANs.jpg" height="148px" style="max-width:100%;"></a>
<a href="/pictures/ILSVRC2012_val_00000537_MGANs.jpg" target="_blank"><img src="/pictures/ILSVRC2012_val_00000537_MGANs.jpg" height="148px" style="max-width:100%;"></a>
</p>

It is our expectation that MGANs will trade quality for speed. The question is: how much? Here are some comparisons between the result of MDANs and MGANs:

<p>
<a href="/pictures/MDANvsMGAN/ILSVRC2012_val_00000500.png" target="_blank"><img src="/pictures/MDANvsMGAN/ILSVRC2012_val_00000500.png" height="170px" style="max-width:100%;"></a>
<a href="/pictures/MDANvsMGAN/ILSVRC2012_val_00000500_MDANs.png" target="_blank"><img src="/pictures/MDANvsMGAN/ILSVRC2012_val_00000500_MDANs.png" height="170px" style="max-width:100%;"></a>
<a href="/pictures/MDANvsMGAN/ILSVRC2012_val_00000500_MGANs.jpg" target="_blank"><img src="/pictures/MDANvsMGAN/ILSVRC2012_val_00000500_MGANs.jpg" height="170px" style="max-width:100%;"></a>
<a href="/pictures/MDANvsMGAN/ILSVRC2012_val_00000501.png" target="_blank"><img src="/pictures/MDANvsMGAN/ILSVRC2012_val_00000501.png" height="170px" style="max-width:100%;"></a>
<a href="/pictures/MDANvsMGAN/ILSVRC2012_val_00000501_MDANs.png" target="_blank"><img src="/pictures/MDANvsMGAN/ILSVRC2012_val_00000501_MDANs.png" height="170px" style="max-width:100%;"></a>
<a href="/pictures/MDANvsMGAN/ILSVRC2012_val_00000501_MGANs.jpg" target="_blank"><img src="/pictures/MDANvsMGAN/ILSVRC2012_val_00000501_MGANs.jpg" height="170px" style="max-width:100%;"></a>
<a href="/pictures/MDANvsMGAN/ILSVRC2012_val_00000502.png" target="_blank"><img src="/pictures/MDANvsMGAN/ILSVRC2012_val_00000502.png" height="210px" style="max-width:100%;"></a>
<a href="/pictures/MDANvsMGAN/ILSVRC2012_val_00000502_MDANs.png" target="_blank"><img src="/pictures/MDANvsMGAN/ILSVRC2012_val_00000502_MDANs.png" height="210px" style="max-width:100%;"></a>
<a href="/pictures/MDANvsMGAN/ILSVRC2012_val_00000502_MGANs.jpg" target="_blank"><img src="/pictures/MDANvsMGAN/ILSVRC2012_val_00000502_MGANs.jpg" height="210px" style="max-width:100%;"></a>
<a href="/pictures/MDANvsMGAN/ILSVRC2012_val_00000503.png" target="_blank"><img src="/pictures/MDANvsMGAN/ILSVRC2012_val_00000503.png" height="170px" style="max-width:100%;"></a>
<a href="/pictures/MDANvsMGAN/ILSVRC2012_val_00000503_MDANs.png" target="_blank"><img src="/pictures/MDANvsMGAN/ILSVRC2012_val_00000503_MDANs.png" height="170px" style="max-width:100%;"></a>
<a href="/pictures/MDANvsMGAN/ILSVRC2012_val_00000503_MGANs.jpg" target="_blank"><img src="/pictures/MDANvsMGAN/ILSVRC2012_val_00000503_MGANs.jpg" height="170px" style="max-width:100%;"></a>
<a href="/pictures/MDANvsMGAN/ILSVRC2012_val_00000507.png" target="_blank"><img src="/pictures/MDANvsMGAN/ILSVRC2012_val_00000507.png" height="170px" style="max-width:100%;"></a>
<a href="/pictures/MDANvsMGAN/ILSVRC2012_val_00000507_MDANs.png" target="_blank"><img src="/pictures/MDANvsMGAN/ILSVRC2012_val_00000507_MDANs.png" height="170px" style="max-width:100%;"></a>
<a href="/pictures/MDANvsMGAN/ILSVRC2012_val_00000507_MGANs.jpg" target="_blank"><img src="/pictures/MDANvsMGAN/ILSVRC2012_val_00000507_MGANs.jpg" height="170px" style="max-width:100%;"></a>
</p>

In general MDANs (middle) give more stylished results, and does a much better job at homegenous background areas (the last two cases). But sometimes MGANs (right) is able to produce comparable results (the first two). 

And MGANs run at least two orders of magnitudes faster. 

# Final remark
There are concurrent works that try to make deep texture synthesis faster. For example, [Ulyanov et al.](https://github.com/DmitryUlyanov/texture_nets) and [Johnson et al.](http://arxiv.org/abs/1603.08155) also achieved significant speed up and very nice results with a feed-forward architecture. Both of these two methods used the gaussian statsitsics constraint proposed by [Gatys et al.](http://arxiv.org/abs/1508.06576). We believe our method is a good complementary: by changing the gaussian statistics constraint to discrimnative networks trained with Markovian patches, it is possible to model more complex texture manifolds (see discussion in our paper).

Last, here are some prelimiary results of training a MGANs for photorealistic synthesis. It learns from 200k face images from [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). The network then transfers [VGG_19](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) encoding (layer ReLU5_1) of new face images (left) into something interesting (right). The synthesized faces have the same poses/layouts as the input faces, but look like different persons :-)

<p>
<a href="/pictures/Face/GT.png" target="_blank"><img src="/pictures/Face/GT.png" height="312px" style="max-width:100%;"></a>
<a href="/pictures/Face/Syn.png" target="_blank"><img src="/pictures/Face/Syn.png" height="312px" style="max-width:100%;"></a>
</p>


# Acknowledgement
* We thank Soumith Chintala for sharing his implementation of [Deep Convolutional Generative Adversarial Networks](https://github.com/soumith/dcgan.torch). 
* We thank the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) team and the [ImageNet](http://image-net.org/) team for sharing their dataset. 
