-- a script for texture synthesis With Markovian Decovolutional Adversarial Networks
local function run_MDAN(params)
  local flag_state = 1
  
  local opt = {}
  opt.dataset_name = params.dataset_name
  opt.stand_imageSize_syn = params.stand_imageSize_syn
  opt.stand_imageSize_example = params.stand_imageSize_example
  opt.flag_pretrained = params.flag_pretrained
  opt.input_content_folder = params.input_content_folder
  opt.output_style_folder = params.output_style_folder
  opt.output_model_folder = params.output_model_folder
  opt.numEpoch = params.numEpoch
  opt.numIterPerEpoch = params.numIterPerEpoch
  opt.contrast_std = params.contrast_std

  os.execute('mkdir ' .. opt.dataset_name .. opt.input_content_folder)
  os.execute('mkdir ' .. opt.dataset_name .. opt.output_style_folder)
  os.execute('mkdir ' .. opt.dataset_name .. opt.output_model_folder)
  opt.target_folder_name = 'Style'
  opt.target_folder = opt.dataset_name .. opt.target_folder_name
  opt.source_folder = opt.dataset_name .. opt.input_content_folder

  opt.stand_atom = params.stand_atom -- make sure image size can be divided by 8
  opt.nc = 3         -- number of channels for color image (fixed to 3)

  -- network
  opt.netS_num = 1
  opt.netS_weights = {1}
  opt.netS_vgg_Outputlayer = {13} -- plus one due to tv layer
  opt.netS_vgg_nOutputPlane = {256}
  opt.netS_vgg_Outputpatchsize = {8} -- value is decided by the design of netS
  opt.netS_vgg_Outputpatchstep = {4}

  opt.netC_num = 1 
  opt.netC_weights = {1} 
  opt.netC_vgg_Outputlayer = {31} -- plus one due to tv layer
  
  opt.tv_weight = 1e-4

  -- data augmentation
  opt.aug_step_rotation = math.pi/18
  opt.aug_step_scale = 1.1
  opt.aug_num_rotation = 1
  opt.aug_num_scale = 1
  opt.aug_flag_flip = true

  -- optimization
  opt.optimizer = 'adam'
  opt.netD_lr = 0.02 -- netD initial learning rate for adam
  opt.netG_lr = 0.02 -- netG initial learning rate for adam
  opt.netD_beta1 = 0.5 -- netD momentum term of adam
  opt.netG_beta1 = 0.5 -- netG momentum term of adam
  opt.real_label = 1          -- value of real label (fixed to 1)
  opt.fake_label = -1         -- value of fake label (fixed to -1, we use max margin instead of BCE)

  -- misc
  opt.display = 1
  opt.gpu = 1               -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
  opt.vgg_proto_file = '../Dataset/model/VGG_ILSVRC_19_layers_deploy.prototxt' 
  opt.vgg_model_file = '../Dataset/model/VGG_ILSVRC_19_layers.caffemodel'
  opt.vgg_backend = 'nn'
  opt.vggD_num_layer = 36


  local weight_sum = 0
  for i_netC = 1, opt.netC_num do
    weight_sum = weight_sum + opt.netC_weights[i_netC]
  end
  for i_netS = 1, opt.netS_num do
    weight_sum = weight_sum + opt.netS_weights[i_netS]
  end
  for i_netC = 1, opt.netC_num do
    opt.netC_weights[i_netC] = opt.netC_weights[i_netC] / weight_sum
  end
  for i_netS = 1, opt.netS_num do
    opt.netS_weights[i_netS] = opt.netS_weights[i_netS] / weight_sum
  end
  opt.manualSeed = torch.random(1, 10000) -- fix seed
  torch.manualSeed(opt.manualSeed)

  --------------------------------
  -- build networks
  --------------------------------
  -- build netVGG
  local net_ = loadcaffe_wrap.load(opt.vgg_proto_file, opt.vgg_model_file, opt.vgg_backend, opt.vggD_num_layer)
  local netVGGD = nn.Sequential()
  netVGGD:add(nn.TVLoss(opt.tv_weight))
  for i_layer = 1, opt.vggD_num_layer do
    netVGGD:add(net_:get(i_layer))
  end
  net_ = nil
  collectgarbage()
  print('netVGGD has been built')
  print(netVGGD)
  netVGGD = util.cudnn(netVGGD)
  netVGGD:cuda()

  local netS = {}
  local netSVGG = {}
  local netC = {}

  for i_netS = 1, opt.netS_num do 
    table.insert(netSVGG, nn.Sequential())
    for i_layer = 1, opt.netS_vgg_Outputlayer[i_netS] do
      netSVGG[i_netS]:add(netVGGD:get(i_layer))
    end   
    print(string.format('netSVGG[%d] has been built', i_netS))
    print(netSVGG[i_netS])     
  end

  if opt.flag_pretrained then 
    for i_netS = 1, opt.netS_num do 
      table.insert(netS, util.load(opt.dataset_name .. opt.output_model_folder .. 'netS_' .. i_netS .. '.t7', opt.gpu))
      print(string.format('netS[%d] has been loaded', i_netS))
      print(netS[i_netS])       
    end
  else
    for i_netS = 1, opt.netS_num do 
      table.insert(netS, nn.Sequential())
      netS[i_netS]:add(nn.LeakyReLU(0.2, true))

      netS[i_netS]:add(nn.SpatialConvolution(opt.netS_vgg_nOutputPlane[i_netS], opt.netS_vgg_nOutputPlane[i_netS], 3, 3, 1, 1, 1, 1)) 
      netS[i_netS]:add(nn.SpatialBatchNormalization(opt.netS_vgg_nOutputPlane[i_netS])):add(nn.LeakyReLU(0.2, true))

      netS[i_netS]:add(nn.SpatialConvolution(opt.netS_vgg_nOutputPlane[i_netS], 1, opt.netS_vgg_Outputpatchsize[i_netS], opt.netS_vgg_Outputpatchsize[i_netS])) -- 1
      netS[i_netS]:add(nn.View(1):setNumInputDims(3))
      netS[i_netS]:apply(weights_init)
      print(string.format('netS[%d] has been built', i_netS))
      print(netS[i_netS]) 
    end
  end


  for i_netS = 1, opt.netS_num do 
    netSVGG[i_netS] = util.cudnn(netSVGG[i_netS])
    netSVGG[i_netS]:cuda()  
  end

  for i_netS = 1, opt.netS_num do 
    netS[i_netS] = util.cudnn(netS[i_netS])
    netS[i_netS]:cuda()  
  end

  -- initialize netC
  for i_netC = 1, opt.netC_num do 
    table.insert(netC, nn.Sequential())
    for i_layer = 1, opt.netC_vgg_Outputlayer[i_netC] do
      netC[i_netC]:add(netVGGD:get(i_layer))
    end
    print(string.format('netC[%d] has been built', i_netC))
    print(netC[i_netC]) 
  end
  for i_netC = 1, opt.netC_num do 
    netC[i_netC] = util.cudnn(netC[i_netC])
    netC[i_netC]:cuda()  
  end

  -- initialize criterion_netC
  local criterion_netC = {}
  for i_netC = 1, opt.netC_num do
    table.insert(criterion_netC, nn.MSECriterion())
  end
  for i_netC = 1, opt.netC_num do
    criterion_netC[i_netC] = criterion_netC[i_netC]:cuda()
  end

  -- initialize criterion_netS
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
  -- build data for target images
  local target_images_names = pl.dir.getallfiles(opt.target_folder, '*.png')
  local num_target_images = #target_images_names
  print(string.format('num of target images: %d', num_target_images))

  -- data augmentation
  local target_copies = {}
  for i_target_image = 1, num_target_images do 
      local image_target = image.load(target_images_names[i_target_image], 3)
      -- resize 
      local max_dim = math.max(image_target:size()[2], image_target:size()[3])
      local scale = opt.stand_imageSize_example / max_dim
      local new_dim_x = math.floor((image_target:size()[3] * scale) / opt.stand_atom) * opt.stand_atom
      local new_dim_y = math.floor((image_target:size()[2] * scale) / opt.stand_atom) * opt.stand_atom
      image_target = image.scale(image_target, new_dim_x, new_dim_y, 'bilinear')
      image_target = image_target:mul(2):add(-1)

      -- make copies
      for i_r = -opt.aug_num_rotation, opt.aug_num_rotation do
        local alpha = opt.aug_step_rotation * i_r 
        local min_x, min_y, max_x, max_y = computeBB(image_target:size()[3], image_target:size()[2], alpha)
        local image_target_rt = image.rotate(image_target, alpha, 'bilinear')
        image_target_rt = image_target_rt[{{1, image_target_rt:size()[1]}, {min_y, max_y}, {min_x, max_x}}]
        for i_s = -opt.aug_num_scale, opt.aug_num_scale do  
          local max_sz = math.floor(math.max(image_target_rt:size()[2], image_target_rt:size()[3]) * torch.pow(opt.aug_step_scale, i_s))
          local target_image_rt_s = image.scale(image_target_rt, max_sz, 'bilinear')
          table.insert(target_copies, target_image_rt_s)
        end
      end

      -- flip
      if opt.aug_flag_flip then
        image_target = image.hflip(image_target)
        for i_r = -opt.aug_num_rotation, opt.aug_num_rotation do
          local alpha = opt.aug_step_rotation * i_r 
          local min_x, min_y, max_x, max_y = computeBB(image_target:size()[3], image_target:size()[2], alpha)
          local image_target_rt = image.rotate(image_target, alpha, 'bilinear')
          image_target_rt = image_target_rt[{{1, image_target_rt:size()[1]}, {min_y, max_y}, {min_x, max_x}}]
          for i_s = -opt.aug_num_scale, opt.aug_num_scale do  
            local max_sz = math.floor(math.max(image_target_rt:size()[2], image_target_rt:size()[3]) * torch.pow(opt.aug_step_scale, i_s))
            local target_image_rt_s = image.scale(image_target_rt, max_sz, 'bilinear')
            table.insert(target_copies, target_image_rt_s)
          end
        end
      end

  end

  for i_target_copies = 1, #target_copies do 
    target_copies[i_target_copies] = target_copies[i_target_copies]:cuda()
  end

  -- collect target neural patches
  local netS_patches = {}
  for i_netS = 1, opt.netS_num do
    -- compute coordinates
    local target_x_per_image = {}
    local target_y_per_image = {}
    local target_imageid

    for i_target_copies = 1, #target_copies do
      local feature_map = netSVGG[i_netS]:forward(target_copies[i_target_copies]):clone()  
      local target_x_, target_y_ = computegrid(feature_map:size()[3], feature_map:size()[2], opt.netS_vgg_Outputpatchsize[i_netS], opt.netS_vgg_Outputpatchstep[i_netS], 1)
      local target_imageid_ = torch.Tensor(target_x_:nElement() * target_y_:nElement()):fill(i_target_copies)

      table.insert(target_x_per_image, torch.Tensor(target_x_:nElement() * target_y_:nElement()):fill(0))
      table.insert(target_y_per_image, torch.Tensor(target_x_:nElement() * target_y_:nElement()):fill(0))

      local count = 1
      for i_row = 1, target_y_:nElement() do
        for i_col = 1, target_x_:nElement() do
          target_x_per_image[i_target_copies][count] = target_x_[i_col]
          target_y_per_image[i_target_copies][count] = target_y_[i_row]
          count = count + 1
        end
      end
      if i_target_copies == 1 then
        target_imageid = target_imageid_:clone()
      else
        target_imageid = torch.cat(target_imageid, target_imageid_, 1)
      end
    end
    
    -- compute patches
    print(string.format('number of target patches: %d', target_imageid:nElement()))
    local target_patches = torch.Tensor(target_imageid:nElement(), opt.netS_vgg_nOutputPlane[i_netS], opt.netS_vgg_Outputpatchsize[i_netS], opt.netS_vgg_Outputpatchsize[i_netS]):cuda()
    local count_patch = 1
    for i_target_copies = 1, #target_copies do
      local feature_map = netSVGG[i_netS]:forward(target_copies[i_target_copies]):clone()    
      for i_patch_per_image = 1, target_x_per_image[i_target_copies]:nElement() do
        target_patches[count_patch] = feature_map[{{}, {target_y_per_image[i_target_copies][i_patch_per_image], target_y_per_image[i_target_copies][i_patch_per_image] + opt.netS_vgg_Outputpatchsize[i_netS] - 1}, {target_x_per_image[i_target_copies][i_patch_per_image], target_x_per_image[i_target_copies][i_patch_per_image] + opt.netS_vgg_Outputpatchsize[i_netS] - 1}}]
        count_patch = count_patch + 1
      end
    end
    table.insert(netS_patches, target_patches)
  end

  local source_images_names = pl.dir.getallfiles(opt.source_folder, '*.png')
  local num_source_images = #source_images_names
  print(string.format('num of source images: %d', num_source_images))

  print('*****************************************************')
  print('Synthesis: ');
  print('*****************************************************') 
  for i_source_image = 1, #source_images_names do

      -- build data for source images
      local image_source = image.load(source_images_names[i_source_image], 3)

      -- resize 
      local max_dim = math.max(image_source:size()[2], image_source:size()[3])
      local scale = opt.stand_imageSize_syn / max_dim
      local new_dim_x = math.floor((image_source:size()[3] * scale) / opt.stand_atom) * opt.stand_atom
      local new_dim_y = math.floor((image_source:size()[2] * scale) / opt.stand_atom) * opt.stand_atom
      image_source = image.scale(image_source, new_dim_x, new_dim_y, 'bilinear')   
      image_source = image_source:mul(2):add(-1)

      image_source = image_source:cuda()
      

      local source_x = {}
      local source_y = {}
      local batchSize = {}

      for i_netS = 1, opt.netS_num do
          local source_x_per_image
          local source_y_per_image

          local feature_map = netSVGG[i_netS]:forward(image_source):clone()
          local source_x_, source_y_ = computegrid(feature_map:size()[3], feature_map:size()[2], opt.netS_vgg_Outputpatchsize[i_netS], opt.netS_vgg_Outputpatchstep[i_netS], 1)
          source_x_per_image = torch.Tensor(source_x_:nElement() * source_y_:nElement()):fill(0)
          source_y_per_image = torch.Tensor(source_x_:nElement() * source_y_:nElement()):fill(0)

          local count = 1
          for i_row = 1, source_y_:nElement() do
              for i_col = 1, source_x_:nElement() do
                source_x_per_image[count] = source_x_[i_col]
                source_y_per_image[count] = source_y_[i_row]
                count = count + 1
              end
          end
          table.insert(source_x, source_x_per_image)
          table.insert(source_y, source_y_per_image)
          table.insert(batchSize, source_x_per_image:nElement())
          print(source_x_per_image:nElement())
      end

      local StyleScore_real = {}
      local StyleScore_G = {}
      local label = {}

      for i_netS = 1, opt.netS_num do
        table.insert(StyleScore_real, torch.Tensor(batchSize[i_netS]))
        table.insert(StyleScore_G, torch.Tensor(batchSize[i_netS]))
        table.insert(label, torch.Tensor(batchSize[i_netS]))
        StyleScore_real[i_netS] = StyleScore_real[i_netS]:cuda()
        StyleScore_G[i_netS] = StyleScore_G[i_netS]:cuda()
        label[i_netS] = label[i_netS]:cuda()
      end

      local errG, errG_Content, errG_Style
      local errS = {}
      for i_netS = 1, opt.netS_num do
        table.insert(errS, 0)
      end
      local epoch_tm = torch.Timer()
      local tm = torch.Timer()
      local data_tm = torch.Timer()

      local parametersS = {}
      local gradParametersS = {}
      for i_netS = 1, opt.netS_num do
        local parametersS_, gradParametersS_ = netS[i_netS]:getParameters()
        table.insert(parametersS, parametersS_)
        table.insert(gradParametersS, gradParametersS_)
      end

      local cur_net_id
      local optimStateG
      local optimStateS = {}
      if opt.optimizer == 'adam' then
        for i_netS = 1, opt.netS_num do
          table.insert(optimStateS, {learningRate = opt.netD_lr, beta1 = opt.netD_beta1,})
        end
        optimStateG =  {learningRate = opt.netG_lr, beta1 = opt.netG_beta1,}
      end

      ---------------------------------------------
      -- some more data preparation
      ---------------------------------------------
      image_G = image_source:clone()
      local netC_feature_map_source = {}
      for i_netC = 1, opt.netC_num do
        local feature_map = netC[i_netC]:forward(image_source):clone()
        table.insert(netC_feature_map_source, feature_map)
      end

      local netC_feature_map_G = {}
      for i_netC = 1, opt.netC_num do
        local feature_map = netC[i_netC]:forward(image_G):clone()
        table.insert(netC_feature_map_G, feature_map)
      end

      local netS_feature_map_G = {}
      for i_netS = 1, opt.netS_num do
        local feature_map = netSVGG[i_netS]:forward(image_source):clone()
        table.insert(netS_feature_map_G, feature_map)
      end

      local netS_selected_patches_real = {}
      for i_netS = 1, opt.netS_num do
        table.insert(netS_selected_patches_real, torch.Tensor(batchSize[i_netS], opt.netS_vgg_nOutputPlane[i_netS], opt.netS_vgg_Outputpatchsize[i_netS], opt.netS_vgg_Outputpatchsize[i_netS]):cuda())
      end
      local netS_selected_patches_G = {}
      for i_netS = 1, opt.netS_num do
        table.insert(netS_selected_patches_G, torch.Tensor(batchSize[i_netS], opt.netS_vgg_nOutputPlane[i_netS], opt.netS_vgg_Outputpatchsize[i_netS], opt.netS_vgg_Outputpatchsize[i_netS]):cuda())
      end

      local gradOutput_S = {}
      for i_netS = 1, opt.netS_num do
        table.insert(gradOutput_S, torch.Tensor(netS_selected_patches_G[i_netS]:size()))
      end

      local netS_gradOutput_S_map = {}
      local netS_gradOutput_S_count = {}

      for i_netS = 1, opt.netS_num do 
        local feature_map = netS_feature_map_G[i_netS]:clone():fill(0)
        table.insert(netS_gradOutput_S_map, feature_map)
      end

      for i_netS = 1, opt.netS_num do
        local feature_map = netS_feature_map_G[i_netS]:clone():fill(0)
        for i_patch = 1, source_x[i_netS]:nElement() do 
            feature_map[{{}, {source_y[i_netS][i_patch], source_y[i_netS][i_patch] + opt.netS_vgg_Outputpatchsize[i_netS] - 1}, {source_x[i_netS][i_patch], source_x[i_netS][i_patch] + opt.netS_vgg_Outputpatchsize[i_netS] - 1}}]:add(1)
        end
        table.insert(netS_gradOutput_S_count, feature_map)
      end

      -- discriminator
      local fDx = function(x)
          netS[cur_net_id]:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
          gradParametersS[cur_net_id]:zero()   

          -- train with real images
          label[cur_net_id]:fill(opt.real_label)
          StyleScore_real[cur_net_id] = netS[cur_net_id]:forward(netS_selected_patches_real[cur_net_id]):clone()
          local err_real = criterion_netS[cur_net_id]:forward(StyleScore_real[cur_net_id], label[cur_net_id])
          local gradInput_StyleScore_real = criterion_netS[cur_net_id]:backward(StyleScore_real[cur_net_id], label[cur_net_id]):clone()  
          netS[cur_net_id]:backward(netS_selected_patches_real[cur_net_id], gradInput_StyleScore_real)
          print(string.format('gradInput_StyleScore_real: %f, gradParametersS: %f', torch.norm(gradInput_StyleScore_real), torch.norm(gradParametersS[cur_net_id])))

          -- train with generated images
          label[cur_net_id]:fill(opt.fake_label)
          StyleScore_G[cur_net_id] = netS[cur_net_id]:forward(netS_selected_patches_G[cur_net_id]):clone()
          local err_fake = criterion_netS[cur_net_id]:forward(StyleScore_G[cur_net_id], label[cur_net_id])
          local gradInput_StyleScore_G = criterion_netS[cur_net_id]:backward(StyleScore_G[cur_net_id], label[cur_net_id]):clone()  
          netS[cur_net_id]:backward(netS_selected_patches_G[cur_net_id], gradInput_StyleScore_G)
          print(string.format('gradInput_StyleScore_G: %f, gradParametersS: %f', torch.norm(gradInput_StyleScore_G), torch.norm(gradParametersS[cur_net_id])))
          
          errS[cur_net_id] = err_real + err_fake
          return errS[cur_net_id], gradParametersS[cur_net_id]
      end

      -- generator
      local fGx = function(x)

          -- gradient from content loss
          local gradOutput_G = image_G:clone():fill(0)
          
          for i_netC = 1, opt.netC_num do
            errG_Content = errG_Content + criterion_netC[i_netC]:forward(netC_feature_map_G[i_netC], netC_feature_map_source[i_netC]) * opt.netC_weights[i_netC]
            local gradOutput_C = criterion_netC[i_netC]:backward(netC_feature_map_G[i_netC], netC_feature_map_source[i_netC])
            netC[i_netC]:forward(image_G)
            local gradInput_C = netC[i_netC]:updateGradInput(image_G, gradOutput_C)
            gradOutput_G = gradOutput_G + gradInput_C:mul(opt.netC_weights[i_netC])
          end

          -- -- gradient from style loss
          for i_netS = 1, opt.netS_num do
            -- update netS_gradOutput_S_map[i_netS][cur_G_id]
            netS_gradOutput_S_map[i_netS]:fill(0)
            for i_patch = 1, source_y[i_netS]:nElement() do
                local ref_oldpatch = netS_gradOutput_S_map[i_netS][{{}, {source_y[i_netS][i_patch], source_y[i_netS][i_patch] + opt.netS_vgg_Outputpatchsize[i_netS] - 1}, {source_x[i_netS][i_patch], source_x[i_netS][i_patch] + opt.netS_vgg_Outputpatchsize[i_netS] - 1}}]
                ref_oldpatch:add(gradOutput_S[i_netS][i_patch])
            end
            netS_gradOutput_S_map[i_netS] = netS_gradOutput_S_map[i_netS]:cdiv(netS_gradOutput_S_count[i_netS])
            netSVGG[i_netS]:forward(image_G)
            local gradInput_S = netSVGG[i_netS]:updateGradInput(image_G, netS_gradOutput_S_map[i_netS])
            gradOutput_G = gradOutput_G + gradInput_S
          end

          errG = errG_Content  + errG_Style 
          return  errG, gradOutput_G
      end

      for epoch = 1, opt.numEpoch do
          epoch_tm:reset()
          local counter = 0
          local im_disp
          for i_iter = 1, opt.numIterPerEpoch do
              tm:reset()

              for i_netC = 1, opt.netC_num do
                netC_feature_map_G[i_netC] = netC[i_netC]:forward(image_G):clone()
              end

              for i_netS = 1, opt.netS_num do
                netS_feature_map_G[i_netS] = netSVGG[i_netS]:forward(image_G):clone()
              end
              
              -- collect generated patches
              for i_netS = 1, opt.netS_num do
                local count_patch = 1
                for i_patch = 1, source_y[i_netS]:nElement() do
                  netS_selected_patches_G[i_netS][count_patch] = netS_feature_map_G[i_netS][{{}, {source_y[i_netS][i_patch], source_y[i_netS][i_patch] + opt.netS_vgg_Outputpatchsize[i_netS] - 1}, {source_x[i_netS][i_patch], source_x[i_netS][i_patch] + opt.netS_vgg_Outputpatchsize[i_netS] - 1}}]:clone()
                  count_patch = count_patch + 1
                end
              end

              -- randomly select real patches
              for i_netS = 1, opt.netS_num do
                local num_patches = netS_patches[i_netS]:size()[1]
                local list_patch = torch.Tensor(batchSize[i_netS])
                for i_patch = 1, batchSize[i_netS] do
                  list_patch[i_patch] = torch.random(1, num_patches)
                end
                for i_patch = 1, batchSize[i_netS] do
                  netS_selected_patches_real[i_netS][i_patch] = netS_patches[i_netS][list_patch[i_patch]]:clone()
                end
              end

              -- train netS
              for i_netS = 1, opt.netS_num do
                cur_net_id = i_netS
                if opt.optimizer == 'adam' then
                  optim.adam(fDx, parametersS[cur_net_id], optimStateS[cur_net_id])
                end
              end

              errG_Content = 0
              errG_Style = 0
              for i_netS = 1, opt.netS_num do
                netS[i_netS]:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
                gradParametersS[i_netS]:zero()  
                label[i_netS]:fill(opt.real_label)
                StyleScore_G[i_netS] = netS[i_netS]:forward(netS_selected_patches_G[i_netS]):clone()
                errS[i_netS] = criterion_netS[i_netS]:forward(StyleScore_G[i_netS], label[i_netS]) 
                errG_Style = errG_Style + errS[i_netS] 
                local gradInput_StyleScore_G = criterion_netS[i_netS]:backward(StyleScore_G[i_netS], label[i_netS]):clone()  
                gradOutput_S[i_netS] = netS[i_netS]:updateGradInput(netS_selected_patches_G[i_netS], gradInput_StyleScore_G)
                print(string.format('gradInput_StyleScore_G: %f, gradOutput_G: %f', torch.norm(gradInput_StyleScore_G), torch.norm(gradOutput_S[i_netS])))
              end
              
              -- update synthesis image
              if opt.optimizer == 'adam' then
                optim.adam(fGx, image_G, optimStateG)
              end    

            -- display & save image
            im_disp = image_G:clone()
            local mean_im_disp = torch.mean(im_disp)
            local std_im_disp = torch.std(im_disp)
            im_disp:add(-mean_im_disp)
            im_disp[torch.lt(im_disp, -std_im_disp * opt.contrast_std)] = -std_im_disp * opt.contrast_std
            im_disp[torch.gt(im_disp, std_im_disp * opt.contrast_std)] = std_im_disp * opt.contrast_std        
            local min_im_disp = torch.min(im_disp)
            local max_im_disp = torch.max(im_disp)
            im_disp:add(-min_im_disp):div(max_im_disp - min_im_disp) 
            if opt.display then       
              disp.image(im_disp, {win=1, title = 'MDAN'})
            end

            -- logging
            print(('Epoch: [%d][%8d / %8d]\t'
                   .. ' errG_Content: %.4f  errG_Style: %.4f'):format(
                 epoch, i_iter,
                 opt.numIterPerEpoch,
                 errG_Content and errG_Content or -1, errG_Style and errG_Style or -1))

          end -- for i_iter = 1, opt.numIterPerEpoch do
      end -- for epoch = 1, opt.numEpoch do

      local im_out = image_G:clone()
      local mean_im_out = torch.mean(im_out)
      local std_im_out = torch.std(im_out)
      im_out:add(-mean_im_out)
       im_out[torch.lt(im_out, -std_im_out * opt.contrast_std)] = -std_im_out * opt.contrast_std
       im_out[torch.gt(im_out, std_im_out * opt.contrast_std)] = std_im_out * opt.contrast_std
      local min_im_out = torch.min(im_out)
      local max_im_out = torch.max(im_out)
      im_out:add(-min_im_out):div(max_im_out - min_im_out) 
      image.save(opt.dataset_name .. opt.output_style_folder .. source_images_names[i_source_image]:match( "([^/]+)$" ), im_out)

      -- clear memory
      parametersS = nil
      gradParametersS = nil
      optimStateG = nil
      optimStateS = nil
      image_source = nil
      image_G = nil
      netC_feature_map_source = nil
      netC_feature_map_G = nil
      netS_feature_map_G = nil
      netS_selected_patches_real = nil
      netS_selected_patches_G = nil
      gradOutput_S = nil
      netS_gradOutput_S_map = nil
      netS_gradOutput_S_count = nil
      im_out = nil
      collectgarbage()
      collectgarbage()

  end -- for i_source_copies = 1, #source_copies do

  for i_netS = 1, opt.netS_num do 
    util.save(opt.dataset_name .. opt.output_model_folder .. 'netS_' .. i_netS .. '.t7', netS[i_netS], opt.gpu)
  end  

  netVGGD = nil 
  for i_netS = 1, opt.netS_num do 
    netS[i_netS] = nil
  end  
  for i_netS = 1, opt.netS_num do 
    netSVGG[i_netS] = nil
  end
  for i_netC = 1, opt.netC_num do 
    netC[i_netC] = nil
  end

  collectgarbage()
  collectgarbage()

  return flag_state
end

return {
  state = run_MDAN
}










