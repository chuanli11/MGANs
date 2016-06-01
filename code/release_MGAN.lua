-- a script for releasing the generator: it basically concatenates the encoder part of VGG to the generator.
require 'torch'
require 'nn'
require 'optim'
require 'cutorch'
require 'cunn'
require 'image'
loadcaffe_wrap = paths.dofile('lib/loadcaffe_wrapper.lua')
util = paths.dofile('lib/util.lua')

local opt = {}
opt.model_name = '../Dataset/VG_Alpilles_ImageNet100/MGAN/epoch_5_netG.t7'
opt.release_name = '../Dataset/VG_Alpilles_ImageNet100/VG_Alpilles_ImageNet100_epoch5.t7'

---*********************************************************************************************************************
-- DO NOT CHANGE AFTER THIS LINE
---*********************************************************************************************************************
opt.vgg_proto_file = '../Dataset/model/VGG_ILSVRC_19_layers_deploy.prototxt'
opt.vgg_model_file = '../Dataset/model/VGG_ILSVRC_19_layers.caffemodel'
opt.vgg_backend = 'nn'
opt.netEnco_vgg_Outputlayer = 21
opt.gpu = 0

local net_vgg = loadcaffe_wrap.load(opt.vgg_proto_file, opt.vgg_model_file, opt.vgg_backend, opt.netEnco_vgg_Outputlayer)
local net_deco = util.load(opt.model_name, opt.gpu)
print(net_deco)

local net_release = nn.Sequential()
for i_layer = 1, opt.netEnco_vgg_Outputlayer do
    net_release:add(net_vgg:get(i_layer))
end  

for i_layer = 1, net_deco:size() do
    net_release:add(net_deco:get(i_layer))
end  

print(net_release)
util.save(opt.release_name, net_release, opt.gpu)

