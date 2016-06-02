-- a script for texture synthesis With Markovian Decovolutional Adversarial Networks (MDAN)
require 'torch'
require 'nn'
require 'optim'
require 'cutorch'
require 'cunn'
require 'image'

disp = require 'display'
pl = require('pl.import_into')()
loadcaffe_wrap = paths.dofile('lib/loadcaffe_wrapper.lua')
util = paths.dofile('lib/util.lua')
paths.dofile('lib/helper.lua')
paths.dofile('lib/tv.lua')
paths.dofile('lib/tv2.lua')
cutorch.setDevice(1)
MDAN_wrapper = require 'MDAN_wrapper'
AG_wrapper = require 'AG_wrapper'
MGAN_wrapper = require 'MGAN_wrapper'


-- general parameters
dataset_name = 'VG_Alpilles_ImageNet100' -- path to dataset
stand_imageSize_syn = 384 -- the largest dimension for the synthesized image
stand_imageSize_example = 384 -- the largest dimension for the style image
stand_atom = 8 -- make sure the image size can be divided by stand_atom
train_imageSize = 128 -- the actural size for training images. 
flag_MDAN = true
flag_AG = true
flag_MGAN = true

-- MDAN parameters
MDAN_numEpoch = 5 -- number of epoch 
MDAN_numIterPerEpoch = 25 -- number of iterations per epoch
MDAN_contrast_std = 2 -- cut off value for controling contrast in the output image. Anyvalue that is outside of (-std_im_out * contrast_std, std_im_out * contrast_std) will be set to -std_im_out * contrast_std or std_im_out * contrast_std
local MDAN_experiments = {
    -- MDAN_experiments[1]: input folder for content images 
    -- MDAN_experiments[2]: output folder for stylized images
    -- MDAN_experiments[3]: flag to use pre-trained discriminative network. In practice we need time to build up the discriminator, so it is better to re-use the dicriminator that has already been trained for a while.
    {'ContentInitial/', 'StyleInitial/', false}, -- run with randomly initialized discriminator
    {'ContentTrain/', 'StyleTrain/', true}, -- run with previously saved discriminator
    {'ContentTest/', 'StyleTest/', true}, -- run with previously saved discriminator
}


-- Augment parameters
AG_sampleStep = 64 -- the sampling step for training images
AG_step_rotation = math.pi/18 -- the step for rotation
AG_step_scale = 1.1 -- the step for scaling
AG_num_rotation = 1 -- number of rotations
AG_num_scale = 1 -- number of scalings
AG_flag_flip = true -- flip image or not


-- MGAN parameters
MGAN_netS_weight = 1e-2  --higher weight for discriminator gives sharper texture, but might deviate from image content
local MGAN_experiments = {
    -- MGAN_experiments[1]: starting epoch. Use value larger than one to load a previously saved model (start_epoch - 1)
    -- MGAN_experiments[2]: ending epoch.
    {1, 5}, -- learn five epochs
}


---*********************************************************************************************************************
-- DO NOT CHANGE AFTER THIS LINE
---*********************************************************************************************************************

------------------------------------------------------------------------------------------------------------------------
-- RUN MDAN
------------------------------------------------------------------------------------------------------------------------
if flag_MDAN then
    for i_test = 1, #MDAN_experiments do
        local MDAN_params = {}
        MDAN_params.dataset_name = '../Dataset/' .. dataset_name .. '/'
        MDAN_params.stand_imageSize_syn = stand_imageSize_syn
        MDAN_params.stand_imageSize_example = stand_imageSize_example
        MDAN_params.stand_atom = stand_atom
        MDAN_params.input_content_folder = MDAN_experiments[i_test][1] 
        MDAN_params.output_style_folder = MDAN_experiments[i_test][2]
        MDAN_params.output_model_folder = 'MDAN/'
        MDAN_params.numEpoch = MDAN_numEpoch
        MDAN_params.numIterPerEpoch = MDAN_numIterPerEpoch
        MDAN_params.contrast_std = MDAN_contrast_std
        MDAN_params.flag_pretrained = MDAN_experiments[i_test][3]
        local MDAN_state = MDAN_wrapper.state(MDAN_params)
        collectgarbage()
    end
end

-- ------------------------------------------------------------------------------------------------------------------------
-- -- RUN Data Augmentation
-- ------------------------------------------------------------------------------------------------------------------------
if flag_AG then
    local AG_params = {}
    AG_params.dataset_name = '../Dataset/' .. dataset_name .. '/'
    AG_params.stand_imageSize_syn = stand_imageSize_syn
    AG_params.stand_atom = stand_atom
    AG_params.AG_imageSize = train_imageSize 
    AG_params.AG_sampleStep = AG_sampleStep 
    AG_params.AG_step_rotation = AG_step_rotation 
    AG_params.AG_step_scale = AG_step_scale 
    AG_params.AG_num_rotation = AG_num_rotation 
    AG_params.AG_num_scale = AG_num_scale 
    AG_params.AG_flag_flip = AG_flag_flip 
    local AG_state = AG_wrapper.state(AG_params)
    collectgarbage()
end

-- ------------------------------------------------------------------------------------------------------------------------
-- -- RUN MGAN
-- ------------------------------------------------------------------------------------------------------------------------
if flag_MGAN then
    for i_test = 1, #MGAN_experiments do
        local MGAN_params = {}
        MGAN_params.dataset_name = '../Dataset/' .. dataset_name .. '/'
        MGAN_params.start_epoch = MGAN_experiments[i_test][1]
        MGAN_params.numEpoch = MGAN_experiments[i_test][2]
        MGAN_params.stand_atom = stand_atom
        MGAN_params.pixel_blockSize = train_imageSize
        MGAN_params.netS_weight = MGAN_netS_weight
        local MGAN_state = MGAN_wrapper.state(MGAN_params)
        collectgarbage()    
    end
end


do return end

