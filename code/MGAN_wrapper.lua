-- a script for texture synthesis With Markovian Generative Adversarial Networks

local function run_MGAN(params)
  local flag_state = 1
  local opt = {}
  opt.dataset_name = params.dataset_name 
  opt.start_epoch = params.start_epoch 
  opt.numEpoch = params.numEpoch 
  opt.stand_atom = params.stand_atom
  opt.pixel_blockSize = params.pixel_blockSize 

  -- data    
  opt.source_folder_name = 'ContentTrainPatch128'
  opt.target_folder_name = 'StyleTrainPatch128'
  opt.testsource_folder_name = 'ContentTestPatch128'
  opt.testtarget_folder_name = 'StyleTestPatch128'  
  opt.experiment_name = 'MGAN/' 

  opt.nc = 3  -- number of channels for color image (fixed to 3)
  opt.nf = 64 -- multiplier for number of feaures at each layer
  opt.pixel_weights = 1 -- higher weight preserves image content, but blurs the results.
  opt.tv_weight = 1e-4 -- higher weight reduces optical noise, but may over-smooth the results.

  -- encoder that produce vgg feature map of an input image. This feature map is used as the input for netG
  opt.netEnco_vgg_Outputlayer = 21
  opt.netEnco_vgg_nOutputPlane = 512 -- this value is decided by netEnco_vgg_Outputlayer
  opt.netEnco_vgg_Outputblocksize = opt.pixel_blockSize / 8 -- the denominator's value is decided by netEnco_vgg_Outputlayer

  -- discriminator
  opt.netS_num = 1
  opt.netS_weights = {params.netS_weight}
  opt.netS_vgg_Outputlayer = {12}
  opt.netS_vgg_nOutputPlane = {256} -- this value is decided by netS_vgg_Outputlayer
  opt.netS_blocksize = {opt.pixel_blockSize / 16} -- the denominator's is decided by the design of netS
  opt.netS_flag_mask = 1 -- flag to mask out patches at the border. This reduce the artefacts of padding.
  opt.netS_border = 1  -- margin to be mask. So only patches between (netS_border + 1, netS_blocksize - netS_border) will be used for back propogation.

  -- optimization
  opt.batchSize = 64 -- number of patches in a batch
  opt.optimizer = 'adam'
  opt.netD_lr = 0.02 -- netD initial learning rate for adam
  opt.netG_lr = 0.02 -- netG initial learning rate for adam
  opt.netD_beta1 = 0.5 -- netD first momentum of adam
  opt.netG_beta1 = 0.5 -- netG first momentum of adam
  opt.real_label = 1 -- value of real label (fixed to 1)
  opt.fake_label = -1 -- value of fake label (fixed to -1)

  -- vgg 
  opt.vgg_proto_file = '../Dataset/model/VGG_ILSVRC_19_layers_deploy.prototxt' 
  opt.vgg_model_file = '../Dataset/model/VGG_ILSVRC_19_layers.caffemodel'
  opt.vgg_backend = 'nn'
  opt.vgg_num_layer = 36

  -- misc
  opt.save_iterval_image = 100   -- save iterval for image
  opt.display = 1            -- display samples while training. 0 = false
  opt.gpu = 1    

  local weight_sum = 0

  weight_sum = weight_sum + opt.pixel_weights
  for i_netS = 1, opt.netS_num do
    weight_sum = weight_sum + opt.netS_weights[i_netS]
  end

  opt.pixel_weights = opt.pixel_weights / weight_sum
  for i_netS = 1, opt.netS_num do
    opt.netS_weights[i_netS] = opt.netS_weights[i_netS] / weight_sum
  end

  os.execute('mkdir ' .. opt.dataset_name .. opt.experiment_name)
  opt.target_folder = opt.dataset_name .. opt.target_folder_name
  opt.source_folder = opt.dataset_name .. opt.source_folder_name
  opt.testsource_folder = opt.dataset_name .. opt.testsource_folder_name
  opt.testtarget_folder = opt.dataset_name .. opt.testtarget_folder_name
  opt.manualSeed = torch.random(1, 10000) -- fix seed
  torch.manualSeed(opt.manualSeed)

  --------------------------------
  -- build networks
  --------------------------------
  -- build netVGG
  local net_ = loadcaffe_wrap.load(opt.vgg_proto_file, opt.vgg_model_file, opt.vgg_backend, opt.vggD_num_layer)
  local netVGG = nn.Sequential()
  for i_layer = 1, opt.vgg_num_layer do
    netVGG:add(net_:get(i_layer))
  end
  net_ = nil
  collectgarbage()
  print('netVGG has been built')
  print(netVGG)
  netVGG = util.cudnn(netVGG)
  netVGG:cuda()

  -- build encoder
  local netEnco = nn.Sequential()
  for i_layer = 1, opt.netEnco_vgg_Outputlayer do
   netEnco:add(netVGG:get(i_layer))
  end  
  print(string.format('netEnco has been built'))
  print(netEnco)   
  netEnco = util.cudnn(netEnco)
  netEnco:cuda()

  -- build generator
  local netG = nn.Sequential()
  if opt.start_epoch > 1 then
    netG = util.load(opt.dataset_name .. opt.experiment_name .. 'epoch_' .. opt.start_epoch - 1 .. '_netG.t7', opt.gpu)
    print(string.format('netG has been loaded'))
    print(netG)
  else
    -- layer 21
    netG:add(nn.SpatialFullConvolution(opt.netEnco_vgg_nOutputPlane, opt.nf * 8, 3, 3, 1, 1, 1, 1)) -- x 1
    netG:add(nn.SpatialBatchNormalization(opt.nf * 8)):add(nn.ReLU(true))
    netG:add(nn.SpatialFullConvolution(opt.nf * 8, opt.nf * 4, 4, 4, 2, 2, 1, 1)) -- x 2
    netG:add(nn.SpatialBatchNormalization(opt.nf * 4)):add(nn.ReLU(true))
    netG:add(nn.SpatialFullConvolution(opt.nf * 4, opt.nf * 2, 4, 4, 2, 2, 1, 1)) -- x 4
    netG:add(nn.SpatialBatchNormalization(opt.nf * 2)):add(nn.ReLU(true))
    netG:add(nn.SpatialFullConvolution(opt.nf * 2, opt.nc, 4, 4, 2, 2, 1, 1)) -- x 8
    netG:add(nn.Tanh())
    netG:apply(weights_init)
    print(string.format('netG has been built'))
    print(netG)  
  end

 
  netG = util.cudnn(netG)
  netG:cuda()

  -- build discriminator
  local netSVGG = {}
  for i_netS = 1, opt.netS_num do 
    table.insert(netSVGG, nn.Sequential())
    -- netSVGG[i_netS]:add(nn.TVLoss2(opt.tv_weight))
    for i_layer = 1, opt.netS_vgg_Outputlayer[i_netS] do
      netSVGG[i_netS]:add(netVGG:get(i_layer))
    end   
    print(string.format('netSVGG[%d] has been built', i_netS))
    print(netSVGG[i_netS])     
  end
  for i_netS = 1, opt.netS_num do 
    netSVGG[i_netS] = util.cudnn(netSVGG[i_netS])
    netSVGG[i_netS]:cuda()  
  end

  local netS = {}
  if opt.start_epoch > 1 then
    for i_netS = 1, opt.netS_num do 
      table.insert(netS, util.load(opt.dataset_name .. opt.experiment_name .. 'epoch_' .. opt.start_epoch - 1 .. '_netS_' .. i_netS .. '.t7', opt.gpu))
      print(string.format('netS[%d] has been loaded', i_netS))
      print(netS[i_netS]) 
    end
  else
    for i_netS = 1, opt.netS_num do
      table.insert(netS, nn.Sequential())
      netS[i_netS]:add(nn.LeakyReLU(0.2, true))
      netS[i_netS]:add(nn.SpatialConvolution(opt.netS_vgg_nOutputPlane[i_netS], opt.nf * 4, 4, 4, 2, 2, 1, 1)) -- x 1/2
      netS[i_netS]:add(nn.SpatialBatchNormalization(opt.nf * 4)):add(nn.LeakyReLU(0.2, true))
      netS[i_netS]:add(nn.SpatialConvolution(opt.nf * 4, opt.nf * 8, 4, 4, 2, 2, 1, 1)) -- x 1/4
      netS[i_netS]:add(nn.SpatialBatchNormalization(opt.nf * 8)):add(nn.LeakyReLU(0.2, true))     
      netS[i_netS]:add(nn.SpatialConvolution(opt.nf * 8, 1, 1, 1)) -- classify each neural patch using convolutional operation
      netS[i_netS]:add(nn.Reshape(opt.batchSize * opt.netS_blocksize[i_netS] * opt.netS_blocksize[i_netS], 1, 1, 1, false)) -- reshape the classification result for computing loss
      netS[i_netS]:add(nn.View(1):setNumInputDims(3))
      netS[i_netS]:apply(weights_init)
      print(string.format('netS[%d] has been built', i_netS))
      print(netS[i_netS]) 
    end
  end
  

  for i_netS = 1, opt.netS_num do 
    netS[i_netS] = util.cudnn(netS[i_netS])
    netS[i_netS]:cuda()  
  end

  local criterion_Pixel = nn.MSECriterion()
  criterion_Pixel:cuda()

  local criterion_netS = {}
  for i_netS = 1, opt.netS_num do
    table.insert(criterion_netS, nn.MarginCriterion(1))
  end
  for i_netS = 1, opt.netS_num do
    criterion_netS[i_netS] = criterion_netS[i_netS]:cuda()
  end

  --------------------------------
  -- build data
  --------------------------------
  local source_images_names = pl.dir.getallfiles(opt.source_folder, '*.png')
  local num_source_images = #source_images_names
  print(string.format('num of source images: %d', num_source_images))

  local target_images_names = pl.dir.getallfiles(opt.target_folder, '*.png')
  local num_target_images = #target_images_names
  print(string.format('num of target images: %d', num_target_images))

  local test_images_names = pl.dir.getallfiles(opt.testsource_folder, '*.png')
  local num_test_images = #test_images_names
  print(string.format('num of test images: %d', num_test_images))

  local BlockPixel_target = torch.Tensor(opt.batchSize, opt.nc, opt.pixel_blockSize, opt.pixel_blockSize)
  BlockPixel_target = BlockPixel_target:cuda();  

  local BlockPixel_source = torch.Tensor(opt.batchSize, opt.nc, opt.pixel_blockSize, opt.pixel_blockSize)
  BlockPixel_source = BlockPixel_source:cuda(); 

  local BlockPixel_G = torch.Tensor(opt.batchSize, opt.nc, opt.pixel_blockSize, opt.pixel_blockSize)
  BlockPixel_G = BlockPixel_G:cuda()

  local BlockPixel_testsource = torch.Tensor(opt.batchSize, opt.nc, opt.pixel_blockSize, opt.pixel_blockSize)
  BlockPixel_testsource = BlockPixel_testsource:cuda();  

  local BlockPixel_testtarget = torch.Tensor(opt.batchSize, opt.nc, opt.pixel_blockSize, opt.pixel_blockSize)
  BlockPixel_testtarget = BlockPixel_testtarget:cuda(); 

  local BlockPixel_Gtest = torch.Tensor(opt.batchSize, opt.nc, opt.pixel_blockSize, opt.pixel_blockSize)
  BlockPixel_Gtest = BlockPixel_Gtest:cuda();  

  local BlockInterface = torch.Tensor(opt.batchSize, opt.netEnco_vgg_nOutputPlane, opt.netEnco_vgg_Outputblocksize, opt.netEnco_vgg_Outputblocksize)
  BlockInterface = BlockInterface:cuda()

  local BlockInterfacetest = torch.Tensor(opt.batchSize, opt.netEnco_vgg_nOutputPlane, opt.netEnco_vgg_Outputblocksize, opt.netEnco_vgg_Outputblocksize)
  BlockInterfacetest = BlockInterfacetest:cuda()

  local BlockVGG_target = {}
  for i_netS = 1, opt.netS_num do
    local feature_map = netSVGG[i_netS]:forward(BlockPixel_target):clone()
    table.insert(BlockVGG_target, feature_map)
  end

  local BlockVGG_G = {}
  for i_netS = 1, opt.netS_num do
    local feature_map = netSVGG[i_netS]:forward(BlockPixel_G):clone()
    table.insert(BlockVGG_G, feature_map)
  end


  local optimStateG = {learningRate = opt.netG_lr, beta1 = opt.netG_beta1,}
  local optimStateS = {}
  for i_netS = 1, opt.netS_num do
    table.insert(optimStateS, {learningRate = opt.netD_lr, beta1 = opt.netD_beta1,})
  end
  optimStateG =  {learningRate = opt.netG_lr, beta1 = opt.netG_beta1,}

  local parametersG, gradparametersG = netG:getParameters()
  local errG = 0
  local errG_Pixel, errG_Style

  local parametersS = {}
  local gradParametersS = {}
  for i_netS = 1, opt.netS_num do
    local parametersS_, gradParametersS_ = netS[i_netS]:getParameters()
    table.insert(parametersS, parametersS_)
    table.insert(gradParametersS, gradParametersS_)
  end

  local list_image_test = torch.Tensor(opt.batchSize)
  for i_img = 1, opt.batchSize do
    list_image_test[i_img] = torch.random(1, num_test_images)
  end    
  for i_img = 1, opt.batchSize do
    BlockPixel_testsource[i_img] = image.load(opt.testsource_folder .. '/' .. list_image_test[i_img] .. '.png', 3)
  end
  BlockPixel_testsource:mul(2):add(-1)
  BlockPixel_testsource = BlockPixel_testsource:cuda()
  BlockInterfacetest = netEnco:forward(BlockPixel_testsource):clone()

  for i_img = 1, opt.batchSize do
    BlockPixel_testtarget[i_img] = image.load(opt.testtarget_folder .. '/' .. list_image_test[i_img] .. '.png', 3)
  end
  BlockPixel_testtarget:mul(2):add(-1)
  BlockPixel_testtarget = BlockPixel_testtarget:cuda()

  local StyleScore_real = {}
  local StyleScore_G = {}
  local label = {}
  for i_netS = 1, opt.netS_num do
    table.insert(StyleScore_real, torch.Tensor(opt.batchSize * opt.netS_blocksize[i_netS] * opt.netS_blocksize[i_netS]))
    table.insert(StyleScore_G, torch.Tensor(opt.batchSize * opt.netS_blocksize[i_netS] * opt.netS_blocksize[i_netS]))
    table.insert(label, torch.Tensor(opt.batchSize * opt.netS_blocksize[i_netS] * opt.netS_blocksize[i_netS]))
    StyleScore_real[i_netS] = StyleScore_real[i_netS]:cuda()
    StyleScore_G[i_netS] = StyleScore_G[i_netS]:cuda()
    label[i_netS] = label[i_netS]:cuda()
  end

  local errS = {}
  for i_netS = 1, opt.netS_num do
    table.insert(errS, 0)
  end

  local netS_mask = {}
  for i_netS = 1, opt.netS_num do
    table.insert(netS_mask, torch.Tensor(opt.batchSize, 1, opt.netS_blocksize[i_netS], opt.netS_blocksize[i_netS]):fill(1))
    if opt.netS_flag_mask == 1 then
      netS_mask[i_netS][{{1, opt.batchSize}, {1, 1}, {opt.netS_border + 1, opt.netS_blocksize[i_netS] - opt.netS_border}, {opt.netS_border + 1, opt.netS_blocksize[i_netS] - opt.netS_border}}]:fill(0)
    end
    netS_mask[i_netS] = netS_mask[i_netS]:reshape(opt.batchSize * opt.netS_blocksize[i_netS] * opt.netS_blocksize[i_netS], 1)
  end
  for i_netS = 1, opt.netS_num do
    netS_mask[i_netS] = netS_mask[i_netS]:cuda()
  end

  local cur_net_id

  -- discriminator
  local fDx = function(x)
    netS[cur_net_id]:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    gradParametersS[cur_net_id]:zero()   

    errS[cur_net_id] = 0

    -- train with real images    
    label[cur_net_id]:fill(opt.real_label)
    
    StyleScore_real[cur_net_id] = netS[cur_net_id]:forward(BlockVGG_target[cur_net_id]):clone()
    local avg_netD_Score_real = torch.sum(StyleScore_real[cur_net_id]) / StyleScore_real[cur_net_id]:nElement()
    if opt.netS_flag_mask == 1 then
      StyleScore_real[cur_net_id][netS_mask[cur_net_id]] = opt.real_label
      avg_netD_Score_real = torch.sum(StyleScore_real[cur_net_id][netS_mask[cur_net_id]:clone():mul(-1):add(1)]) / (StyleScore_real[cur_net_id]:nElement() - torch.sum(netS_mask[cur_net_id]))
    end
    errS[cur_net_id] = errS[cur_net_id] + criterion_netS[cur_net_id]:forward(StyleScore_real[cur_net_id], label[cur_net_id])
    local gradInput_StyleScore_real = criterion_netS[cur_net_id]:backward(StyleScore_real[cur_net_id], label[cur_net_id]):clone()  
    netS[cur_net_id]:backward(BlockVGG_target[cur_net_id], gradInput_StyleScore_real)
    print("netS avg_netD_Score_real: " .. avg_netD_Score_real .. ", gradParametersS: " .. torch.norm(gradParametersS[cur_net_id]))

    label[cur_net_id]:fill(opt.fake_label)
    StyleScore_G[cur_net_id] = netS[cur_net_id]:forward(BlockVGG_G[cur_net_id]):clone()
    local avg_netD_Score_G = torch.sum(StyleScore_G[cur_net_id]) / StyleScore_G[cur_net_id]:nElement()
    if opt.netS_flag_mask == 1 then
      StyleScore_G[cur_net_id][netS_mask[cur_net_id]] = opt.fake_label
      avg_netD_Score_G = torch.sum(StyleScore_G[cur_net_id][netS_mask[cur_net_id]:clone():mul(-1):add(1)]) / (StyleScore_G[cur_net_id]:nElement() - torch.sum(netS_mask[cur_net_id]))
    end
    errS[cur_net_id] = errS[cur_net_id] + criterion_netS[cur_net_id]:forward(StyleScore_G[cur_net_id], label[cur_net_id])
    local gradInput_StyleScore_G = criterion_netS[cur_net_id]:backward(StyleScore_G[cur_net_id], label[cur_net_id]):clone()  
    netS[cur_net_id]:backward(BlockVGG_G[cur_net_id], gradInput_StyleScore_G)
    print("netS avg_netD_Score_G: " .. avg_netD_Score_G .. ", gradParametersS: " .. torch.norm(gradParametersS[cur_net_id]))

    gradParametersS[cur_net_id] = gradParametersS[cur_net_id]:div(opt.batchSize)

    gradInput_StyleScore_real = nil
    gradInput_StyleScore_G = nil
    collectgarbage()
    collectgarbage()

    return errS[cur_net_id], gradParametersS[cur_net_id]
  end


  -- generator
  local fGx = function(x)
    netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    gradparametersG:zero()  
    local gradOutput_G = BlockPixel_G:clone():fill(0)
    errG = 0
    errG_Pixel = 0
    errG_Style = 0

    -- pixel loss
    errG_Pixel = criterion_Pixel:forward(BlockPixel_G, BlockPixel_target)
    local gradOutput_G_Pixel = criterion_Pixel:backward(BlockPixel_G, BlockPixel_target)
    gradOutput_G = gradOutput_G + gradOutput_G_Pixel:mul(opt.pixel_weights)
    errG = errG + errG_Pixel

    -- style loss
    for i_netS = 1, opt.netS_num do
      label[i_netS]:fill(opt.real_label)
      StyleScore_G[i_netS] = netS[i_netS]:forward(BlockVGG_G[i_netS]):clone()
      local avg_netD_Score_G = torch.sum(StyleScore_G[i_netS]) / StyleScore_G[i_netS]:nElement()
      if opt.netS_flag_mask == 1 then
        StyleScore_G[i_netS][netS_mask[i_netS]] = opt.real_label
        avg_netD_Score_G = torch.sum(StyleScore_G[cur_net_id][netS_mask[cur_net_id]:clone():mul(-1):add(1)]) / (StyleScore_G[cur_net_id]:nElement() - torch.sum(netS_mask[cur_net_id]))
      end      
      print("netG avg_netD_Score_G: " .. avg_netD_Score_G)

      errS[i_netS] = criterion_netS[i_netS]:forward(StyleScore_G[i_netS], label[i_netS])
      errG_Style = errG_Style + errS[i_netS]
      local gradInput_StyleScore_G = criterion_netS[i_netS]:backward(StyleScore_G[i_netS], label[i_netS]):clone()  
      local gradInput_netS = netS[i_netS]:updateGradInput(BlockVGG_G[i_netS], gradInput_StyleScore_G)
      netSVGG[i_netS]:forward(BlockPixel_G)
      local gradInput_Style = netSVGG[i_netS]:updateGradInput(BlockPixel_G, gradInput_netS)
      gradOutput_G = gradOutput_G + gradInput_Style:mul(opt.netS_weights[i_netS])

      gradInput_StyleScore_G = nil
      gradInput_netS = nil
      collectgarbage()
      collectgarbage()
    end

    netG:backward(BlockInterface, gradOutput_G)
    print("gradparametersG: " .. torch.norm(gradparametersG))

    errG = errG + errG_Style

    gradOutput_G_Pixel = nil
    gradOutput_G = nil
    collectgarbage()
    collectgarbage()
    return  errG, gradparametersG
  end

  print('*****************************************************')
  print('Training Loop: ');
  print('*****************************************************') 
  local epoch_tm = torch.Timer()
  local tm = torch.Timer()
  local data_tm = torch.Timer()
  local record_err = torch.Tensor(1)
  record_err:fill(criterion_Pixel:forward(BlockPixel_testsource, BlockPixel_testtarget))


  for epoch = opt.start_epoch, opt.numEpoch do
    local source 
    local sourcetest
    local target
    local targettest
    local generated
    local generatedtest

   local counter = 0
   for i_iter = 1, num_source_images, opt.batchSize do
     tm:reset()
     counter = counter + 1
      
      -- randomly select images to train
      local list_image = torch.Tensor(opt.batchSize)
      for i_img = 1, opt.batchSize do
        list_image[i_img] = torch.random(1, num_source_images)
      end  

      -- read PatchPixel_real and PatchPixel_photo
      for i_img = 1, opt.batchSize do
        BlockPixel_target[i_img] = image.load(opt.target_folder .. '/' .. list_image[i_img] .. '.png', 3)
      end
      BlockPixel_target:mul(2):add(-1)
      BlockPixel_target = BlockPixel_target:cuda()

      for i_img = 1, opt.batchSize do
        BlockPixel_source[i_img] = image.load(opt.source_folder .. '/' .. list_image[i_img] .. '.png', 3)
      end
      BlockPixel_source:mul(2):add(-1)
      BlockPixel_source = BlockPixel_source:cuda()

      BlockInterface = netEnco:forward(BlockPixel_source)
      BlockPixel_G = netG:forward(BlockInterface)

     
      for i_netS = 1, opt.netS_num do
        BlockVGG_target[i_netS] = netSVGG[i_netS]:forward(BlockPixel_target):clone()
        BlockVGG_G[i_netS] = netSVGG[i_netS]:forward(BlockPixel_G):clone()
      end

      -- train netS
      for i_netS = 1, opt.netS_num do
        cur_net_id = i_netS
        optim.adam(fDx, parametersS[cur_net_id], optimStateS[cur_net_id])
      end

      -- train netG
      optim.adam(fGx, parametersG, optimStateG)

      if opt.display then
        source = BlockPixel_source
        sourcetest = BlockPixel_testsource
        target = BlockPixel_target
        targettest = BlockPixel_testtarget
        generated = netG:forward(BlockInterface):clone()
        generatedtest = netG:forward(BlockInterfacetest):clone()
        disp.image(source, {win=1, title='source'})
        disp.image(target, {win=2, title='target'})
        disp.image(generated, {win=3, title='generated'})
      end

     if counter == math.floor(num_target_images / opt.batchSize) or counter % opt.save_iterval_image == 0 then
        local img_source = image.toDisplayTensor{input = source, nrow = math.ceil(math.sqrt(source:size(1)))}
        local img_sourcetest = image.toDisplayTensor{input = sourcetest, nrow = math.ceil(math.sqrt(sourcetest:size(1)))}
        local img_target = image.toDisplayTensor{input = target, nrow = math.ceil(math.sqrt(target:size(1)))}
        local img_targettest = image.toDisplayTensor{input = targettest, nrow = math.ceil(math.sqrt(targettest:size(1)))}
        local img_generated = image.toDisplayTensor{input = generated, nrow = math.ceil(math.sqrt(generated:size(1)))}
        local img_generatedtest = image.toDisplayTensor{input = generatedtest, nrow = math.ceil(math.sqrt(generatedtest:size(1)))}

        local img_out = torch.Tensor(img_generated:size()[1], img_generated:size()[2] + img_generated:size()[2] + 16, img_generated:size()[3] + img_generated:size()[3] + img_generated:size()[3] + 32):fill(0):cuda()

        img_out[{{1,img_generated:size()[1]}, {1,img_generated:size()[2]}, {1,img_generated:size(3)}}] = img_source
        img_out[{{1,img_generated:size()[1]}, {1,img_generated:size()[2]}, {img_out:size()[3] - img_generated:size()[3] - 16 - img_generated:size()[3] + 1, img_out:size()[3] - img_generated:size()[3] - 16}}] = img_generated
        img_out[{{1,img_generated:size()[1]}, {1,img_generated:size()[2]}, {img_out:size()[3] - img_generated:size()[3] + 1, img_out:size()[3]}}] = img_target

        img_out[{{1,img_generated:size()[1]}, {img_out:size()[2] - img_generated:size()[2] + 1, img_out:size()[2]}, {1,img_generated:size(3)}}] = img_sourcetest
        img_out[{{1,img_generated:size()[1]}, {img_out:size()[2] - img_generated:size()[2] + 1, img_out:size()[2]}, {img_out:size()[3] - img_generated:size()[3] - 16 - img_generated:size()[3] + 1, img_out:size()[3] - img_generated:size()[3] - 16}}] = img_generatedtest
        img_out[{{1,img_generated:size()[1]}, {img_out:size()[2] - img_generated:size()[2] + 1, img_out:size()[2]}, {img_out:size()[3] - img_generated:size()[3] + 1, img_out:size()[3]}}] = img_targettest

        image.save(opt.dataset_name .. opt.experiment_name .. 'epoch_' .. epoch .. '_' .. counter .. '.png', img_out)
     end 

       -- logging
     print(('Epoch: [%d][%8d / %8d]\t Time: %.3f '
              .. 'errG: %.4f'):format(
            epoch, ((i_iter-1) / opt.batchSize),
            math.floor(num_target_images / opt.batchSize),
            tm:time().real, errG and errG or -1))
   end

    parametersG = nil
    gradparametersG = nil
    util.save(opt.dataset_name .. opt.experiment_name .. 'epoch_' .. epoch .. '_netG.t7', netG, opt.gpu)
    parametersG, gradparametersG = netG:getParameters()

    for i_netS = 1, opt.netS_num do 
      print(opt.dataset_name .. opt.experiment_name .. 'epoch_' .. epoch .. '_netS_' .. i_netS .. '.t7')
      parametersS[i_netS] = nil
      gradParametersS[i_netS] = nil
      util.save(opt.dataset_name .. opt.experiment_name .. 'epoch_' .. epoch .. '_netS_' .. i_netS .. '.t7', netS[i_netS], opt.gpu)
      parametersS[i_netS], gradParametersS[i_netS] = netS[i_netS]:getParameters()
    end  

    record_err = torch.cat(record_err, torch.Tensor(1):fill(criterion_Pixel:forward(generatedtest, targettest)), 1)
    print(record_err)
    disp.plot(torch.cat(torch.linspace(0, record_err:nElement(), record_err:nElement()), record_err, 2), {win=7, title='energy'})
  end -- for epoch = opt.start_epoch, opt.numEpoch do

  netVGG = nil
  for i_netS = 1, opt.netS_num do 
    netS[i_netS] = nil
  end  
  for i_netS = 1, opt.netS_num do 
    netSVGG[i_netS] = nil
  end  
  netEnco = nil
  netG = nil

  collectgarbage()
  collectgarbage()
  
  return flag_state
end

return {
  state = run_MGAN
}






