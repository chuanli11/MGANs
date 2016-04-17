require 'torch'
require 'nn'
require 'optim'
require 'cutorch'
require 'cunn'
require 'image'
loadcaffe_wrap = paths.dofile('lib/loadcaffe_wrapper.lua')
util = paths.dofile('lib/util.lua')
pl = require('pl.import_into')()

local cmd = torch.CmdLine()
local opt = cmd:parse(arg)

opt.max_length = 512 -- change this value for image size. Larger images needs more gpu memory.
opt.noise_weight = 0.2 -- change this weight for balancing between style and content. Stronger noise makes the synthesis more stylish. 

-------------------------------------------------------------------------------------------------------------------
-- do not change after this line
-------------------------------------------------------------------------------------------------------------------
opt.noise_name = 'noise.jpg'
opt.gpu = 1
opt.model_name = '../model/Picasso.t7'
opt.input_folder_name = 'input_photo_imageNet/' 
opt.output_folder_name = 'output/'
opt.stand_atom = 8

os.execute('mkdir ' .. opt.output_folder_name)

local input_images_names = pl.dir.getallfiles(opt.input_folder_name, '*.jpg')
local num_input_images = #input_images_names
local noise_image = image.load(opt.noise_name, 3)
local net = util.load(opt.model_name, opt.gpu)
net:cuda()

print('*****************************************************')
print('Testing: ');
print('*****************************************************') 
local counter = 0
for i_img = 1, num_input_images  do
    -- resize the image image
    local image_input = image.load(input_images_names[i_img], 3)
    local max_dim = math.max(image_input:size()[2], image_input:size()[3])
    local scale = opt.max_length / max_dim
    local new_dim_x = math.floor((image_input:size()[3] * scale) / opt.stand_atom) * opt.stand_atom
    local new_dim_y = math.floor((image_input:size()[2] * scale) / opt.stand_atom) * opt.stand_atom
    image_input = image.scale(image_input, new_dim_x, new_dim_y, 'bilinear')
    
    -- add noise to the image (improve background quality)
    local noise_image_ = image.scale(noise_image, new_dim_x, new_dim_y, 'bilinear') 
    image_input:add(noise_image_:mul(opt.noise_weight))
    image_input:resize(1, image_input:size()[1], image_input:size()[2], image_input:size()[3])
    image_input:mul(2):add(-1)
    image_input = image_input:cuda()

    -- decode image with a single forward prop
	local tm = torch.Timer()
	local image_syn = net:forward(image_input)  
    cutorch.synchronize()
	print(string.format('Image size: %d by %d, time: %f', image_input:size()[3], image_input:size()[4], tm:time().real))
    image_syn = image.toDisplayTensor{input = image_syn, nrow = math.ceil(math.sqrt(image_syn:size(1)))}

    -- save image
    local image_name = input_images_names[i_img]:match("([^/]+)$")
    image_name = string.sub(image_name, 1, string.len(image_name) - 4)
    image.save(opt.output_folder_name .. image_name .. '_MGANs.jpg', image_syn)

    -- clear memory
    image_input = nil
    image_syn = nil
    noise_image_ = nil
    collectgarbage()
end
