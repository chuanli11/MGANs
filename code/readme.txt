We release a few pretrained generative models (in the ../model folder) to stylize photo into different styles.

To use the models, please run "th demo_MGANs.lua" in your terminal. You need to create a input folder (with pictures of “.jpg" format). Corresponding results are saved in the "output" folder. We created an example "input_photo_imageNet" folder with a bunch of randomly selected imagenet photos.

In "demo_MGANs.lua", it is possible to set a different synthesis resolution with the "opt.max_length" parameter. With 2Gb GPU it is possible to run image resolution up to 384-by-384. With 4Gb GPU you can run image resolution of 512-by-512. 

You can balance between content and texture with the "opt.noise_weight" parameter. 

The decoding time for each image is printed in the terminal. Notice the first image takes longer due to the necessity of allocating GPU memory. This only needs to be done once, and the remaining photos will be processed in the normal speed.

Prerequisite:
The script is built upon torch. Currently it is only supported by Ubuntu and Mac.
1) Install cuda (tested with version 7.5, https://developer.nvidia.com/cuda-toolkit)
2) Install torch (http://torch.ch/docs/getting-started.html)
3) Install the image package (in terminal: luarocks install image)
3) Install cutorch (in terminal: luarocks install cutorch)
4) Install cunn (in terminal: luarocks install cunn)
5) You might need to add cuda library to your system path. To do so type "LD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-7.5/lib" in the terminal (Mac), or "LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64" in Ubuntu

Disclaimer: the util.lua file (for loading the pre-saved model) is written by third-party researchers.



