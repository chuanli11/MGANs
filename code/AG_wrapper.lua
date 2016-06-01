local function run_AG(params)
  local flag_state = 1
  dataset_name = params.dataset_name
  stand_imageSize_syn = params.stand_imageSize_syn 
  stand_atom = params.stand_atom
  imageSize = params.AG_imageSize
  sampleStep = params.AG_sampleStep 
  step_rotation = params.AG_step_rotation
  step_scale = params.AG_step_scale
  num_rotation = params.AG_num_rotation
  num_scale = params.AG_num_scale
  flag_flip = params.AG_flag_flip

  -----------------------------------------------------------------
  -- DONOT CHANGE AFTER THIS LINE
  -----------------------------------------------------------------

  -------------------------------------------------------------------------------------------------------------------------------
  -- Generate Testing Images
  -------------------------------------------------------------------------------------------------------------------------------
  data_Content = 'ContentTest/'
  data_Style = 'StyleTest/'
  patch_Content = 'ContentTestPatch' .. imageSize .. '/'
  patch_Style = 'StyleTestPatch' .. imageSize .. '/'

  folder_source = dataset_name .. data_Content
  folder_source_patch = dataset_name .. patch_Content
  folder_target = dataset_name .. data_Style
  folder_target_patch = dataset_name .. patch_Style

  os.execute('mkdir ' .. folder_source_patch)  
  os.execute('mkdir ' .. folder_target_patch)  

  local source_images_names = pl.dir.getallfiles(folder_source, '*.png')
  local num_source_images = #source_images_names
  print(string.format('num of source images: %d', num_source_images))

  count = 1;
  for i_source_image = 1, num_source_images do 
    local image_source = image.load(source_images_names[i_source_image], 3)

    -- resize 
    local max_dim = math.max(image_source:size()[2], image_source:size()[3])
    local scale = stand_imageSize_syn / max_dim
    local new_dim_x = math.floor((image_source:size()[3] * scale) / stand_atom) * stand_atom
    local new_dim_y = math.floor((image_source:size()[2] * scale) / stand_atom) * stand_atom
    image_source = image.scale(image_source, new_dim_x, new_dim_y, 'bilinear')

    -- make copies
    for i_r = -num_rotation, num_rotation do
      local alpha = step_rotation * i_r 
      local min_x, min_y, max_x, max_y = computeBB(image_source:size()[3], image_source:size()[2], alpha)
      local image_source_rt = image.rotate(image_source, alpha, 'bilinear')
      image_source_rt = image_source_rt[{{1, image_source_rt:size()[1]}, {min_y, max_y}, {min_x, max_x}}]
      for i_s = -num_scale, num_scale do  
        local max_sz = math.floor(math.max(image_source_rt:size()[2], image_source_rt:size()[3]) * torch.pow(step_scale, i_s))
        local image_source_rt_s = image.scale(image_source_rt, max_sz, 'bilinear')
        
        for i_row = 1, image_source_rt_s:size()[2] - imageSize + 1, sampleStep do
        	for i_col = 1, image_source_rt_s:size()[3] - imageSize + 1, sampleStep do
              image.save(folder_source_patch .. count .. '.png', image_source_rt_s[{{1, 3}, {i_row, i_row + imageSize - 1}, {i_col, i_col + imageSize - 1}}])
              count = count + 1;
        	end
        end

      end
    end

    -- flip
    if flag_flip then
      image_source = image.hflip(image_source)
      for i_r = -num_rotation, num_rotation do
        local alpha = step_rotation * i_r 
        local min_x, min_y, max_x, max_y = computeBB(image_source:size()[3], image_source:size()[2], alpha)
        local image_source_rt = image.rotate(image_source, alpha, 'bilinear')
        image_source_rt = image_source_rt[{{1, image_source_rt:size()[1]}, {min_y, max_y}, {min_x, max_x}}]
        for i_s = -num_scale, num_scale do  
          local max_sz = math.floor(math.max(image_source_rt:size()[2], image_source_rt:size()[3]) * torch.pow(step_scale, i_s))
          local image_source_rt_s = image.scale(image_source_rt, max_sz, 'bilinear')

  	    for i_row = 1, image_source_rt_s:size()[2] - imageSize + 1, sampleStep do
  	      for i_col = 1, image_source_rt_s:size()[3] - imageSize + 1, sampleStep do
  	          image.save(folder_source_patch .. count .. '.png', image_source_rt_s[{{1, 3}, {i_row, i_row + imageSize - 1}, {i_col, i_col + imageSize - 1}}])
  	          count = count + 1;
  	      end
  	    end
         
        end
      end
    end

  end

local target_images_names = pl.dir.getallfiles(folder_target, '*.png')
local num_target_images = #target_images_names
print(string.format('num of target images: %d', num_target_images))
count = 1;
for i_target_image = 1, num_target_images do 
  local image_target = image.load(target_images_names[i_target_image], 3)

  -- resize 
  local max_dim = math.max(image_target:size()[2], image_target:size()[3])
  local scale = stand_imageSize_syn / max_dim
  local new_dim_x = math.floor((image_target:size()[3] * scale) / stand_atom) * stand_atom
  local new_dim_y = math.floor((image_target:size()[2] * scale) / stand_atom) * stand_atom
  image_target = image.scale(image_target, new_dim_x, new_dim_y, 'bilinear')

  -- make copies
  for i_r = -num_rotation, num_rotation do
    local alpha = step_rotation * i_r 
    local min_x, min_y, max_x, max_y = computeBB(image_target:size()[3], image_target:size()[2], alpha)
    local image_target_rt = image.rotate(image_target, alpha, 'bilinear')
    image_target_rt = image_target_rt[{{1, image_target_rt:size()[1]}, {min_y, max_y}, {min_x, max_x}}]
    for i_s = -num_scale, num_scale do  
      local max_sz = math.floor(math.max(image_target_rt:size()[2], image_target_rt:size()[3]) * torch.pow(step_scale, i_s))
      local image_target_rt_s = image.scale(image_target_rt, max_sz, 'bilinear')
      
      for i_row = 1, image_target_rt_s:size()[2] - imageSize + 1, sampleStep do
      	for i_col = 1, image_target_rt_s:size()[3] - imageSize + 1, sampleStep do
            image.save(folder_target_patch .. count .. '.png', image_target_rt_s[{{1, 3}, {i_row, i_row + imageSize - 1}, {i_col, i_col + imageSize - 1}}])
            count = count + 1;
      	end
      end

    end
  end

  -- flip
  if flag_flip then
    image_target = image.hflip(image_target)
    for i_r = -num_rotation, num_rotation do
      local alpha = step_rotation * i_r 
      local min_x, min_y, max_x, max_y = computeBB(image_target:size()[3], image_target:size()[2], alpha)
      local image_target_rt = image.rotate(image_target, alpha, 'bilinear')
      image_target_rt = image_target_rt[{{1, image_target_rt:size()[1]}, {min_y, max_y}, {min_x, max_x}}]
      for i_s = -num_scale, num_scale do  
        local max_sz = math.floor(math.max(image_target_rt:size()[2], image_target_rt:size()[3]) * torch.pow(step_scale, i_s))
        local image_target_rt_s = image.scale(image_target_rt, max_sz, 'bilinear')

	    for i_row = 1, image_target_rt_s:size()[2] - imageSize + 1, sampleStep do
	      for i_col = 1, image_target_rt_s:size()[3] - imageSize + 1, sampleStep do
	          image.save(folder_target_patch .. count .. '.png', image_target_rt_s[{{1, 3}, {i_row, i_row + imageSize - 1}, {i_col, i_col + imageSize - 1}}])
	          count = count + 1;
	      end
	    end
       
      end
    end
  end

end

-------------------------------------------------------------------------------------------------------------------------------
-- Generate Training Images
-------------------------------------------------------------------------------------------------------------------------------
data_Content = 'ContentTrain/'
data_Style = 'StyleTrain/'
patch_Content = 'ContentTrainPatch' .. imageSize .. '/'
patch_Style = 'StyleTrainPatch' .. imageSize .. '/'

folder_source = dataset_name .. data_Content
folder_source_patch = dataset_name .. patch_Content
folder_target = dataset_name .. data_Style
folder_target_patch = dataset_name .. patch_Style

os.execute('mkdir ' .. folder_source_patch)  
os.execute('mkdir ' .. folder_target_patch)  

local source_images_names = pl.dir.getallfiles(folder_source, '*.png')
local num_source_images = #source_images_names
print(string.format('num of source images: %d', num_source_images))

count = 1;
for i_source_image = 1, num_source_images do 
  local image_source = image.load(source_images_names[i_source_image], 3)

  -- resize 
  local max_dim = math.max(image_source:size()[2], image_source:size()[3])
  local scale = stand_imageSize_syn / max_dim
  local new_dim_x = math.floor((image_source:size()[3] * scale) / stand_atom) * stand_atom
  local new_dim_y = math.floor((image_source:size()[2] * scale) / stand_atom) * stand_atom
  image_source = image.scale(image_source, new_dim_x, new_dim_y, 'bilinear')

  -- make copies
  for i_r = -num_rotation, num_rotation do
    local alpha = step_rotation * i_r 
    local min_x, min_y, max_x, max_y = computeBB(image_source:size()[3], image_source:size()[2], alpha)
    local image_source_rt = image.rotate(image_source, alpha, 'bilinear')
    image_source_rt = image_source_rt[{{1, image_source_rt:size()[1]}, {min_y, max_y}, {min_x, max_x}}]
    for i_s = -num_scale, num_scale do  
      local max_sz = math.floor(math.max(image_source_rt:size()[2], image_source_rt:size()[3]) * torch.pow(step_scale, i_s))
      local image_source_rt_s = image.scale(image_source_rt, max_sz, 'bilinear')
      
      for i_row = 1, image_source_rt_s:size()[2] - imageSize + 1, sampleStep do
      	for i_col = 1, image_source_rt_s:size()[3] - imageSize + 1, sampleStep do
            image.save(folder_source_patch .. count .. '.png', image_source_rt_s[{{1, 3}, {i_row, i_row + imageSize - 1}, {i_col, i_col + imageSize - 1}}])
            count = count + 1;
      	end
      end

    end
  end

  -- flip
  if flag_flip then
    image_source = image.hflip(image_source)
    for i_r = -num_rotation, num_rotation do
      local alpha = step_rotation * i_r 
      local min_x, min_y, max_x, max_y = computeBB(image_source:size()[3], image_source:size()[2], alpha)
      local image_source_rt = image.rotate(image_source, alpha, 'bilinear')
      image_source_rt = image_source_rt[{{1, image_source_rt:size()[1]}, {min_y, max_y}, {min_x, max_x}}]
      for i_s = -num_scale, num_scale do  
        local max_sz = math.floor(math.max(image_source_rt:size()[2], image_source_rt:size()[3]) * torch.pow(step_scale, i_s))
        local image_source_rt_s = image.scale(image_source_rt, max_sz, 'bilinear')

	    for i_row = 1, image_source_rt_s:size()[2] - imageSize + 1, sampleStep do
	      for i_col = 1, image_source_rt_s:size()[3] - imageSize + 1, sampleStep do
	          image.save(folder_source_patch .. count .. '.png', image_source_rt_s[{{1, 3}, {i_row, i_row + imageSize - 1}, {i_col, i_col + imageSize - 1}}])
	          count = count + 1;
	      end
	    end
       
      end
    end
  end

end


local target_images_names = pl.dir.getallfiles(folder_target, '*.png')
local num_target_images = #target_images_names
print(string.format('num of target images: %d', num_target_images))

count = 1;
for i_target_image = 1, num_target_images do 
  local image_target = image.load(target_images_names[i_target_image], 3)

  -- resize 
  local max_dim = math.max(image_target:size()[2], image_target:size()[3])
  local scale = stand_imageSize_syn / max_dim
  local new_dim_x = math.floor((image_target:size()[3] * scale) / stand_atom) * stand_atom
  local new_dim_y = math.floor((image_target:size()[2] * scale) / stand_atom) * stand_atom
  image_target = image.scale(image_target, new_dim_x, new_dim_y, 'bilinear')

  -- make copies
  for i_r = -num_rotation, num_rotation do
    local alpha = step_rotation * i_r 
    local min_x, min_y, max_x, max_y = computeBB(image_target:size()[3], image_target:size()[2], alpha)
    local image_target_rt = image.rotate(image_target, alpha, 'bilinear')
    image_target_rt = image_target_rt[{{1, image_target_rt:size()[1]}, {min_y, max_y}, {min_x, max_x}}]
    for i_s = -num_scale, num_scale do  
      local max_sz = math.floor(math.max(image_target_rt:size()[2], image_target_rt:size()[3]) * torch.pow(step_scale, i_s))
      local image_target_rt_s = image.scale(image_target_rt, max_sz, 'bilinear')
      
      for i_row = 1, image_target_rt_s:size()[2] - imageSize + 1, sampleStep do
      	for i_col = 1, image_target_rt_s:size()[3] - imageSize + 1, sampleStep do
            image.save(folder_target_patch .. count .. '.png', image_target_rt_s[{{1, 3}, {i_row, i_row + imageSize - 1}, {i_col, i_col + imageSize - 1}}])
            count = count + 1;
      	end
      end

    end
  end

  -- flip
  if flag_flip then
    image_target = image.hflip(image_target)
    for i_r = -num_rotation, num_rotation do
      local alpha = step_rotation * i_r 
      local min_x, min_y, max_x, max_y = computeBB(image_target:size()[3], image_target:size()[2], alpha)
      local image_target_rt = image.rotate(image_target, alpha, 'bilinear')
      image_target_rt = image_target_rt[{{1, image_target_rt:size()[1]}, {min_y, max_y}, {min_x, max_x}}]
      for i_s = -num_scale, num_scale do  
        local max_sz = math.floor(math.max(image_target_rt:size()[2], image_target_rt:size()[3]) * torch.pow(step_scale, i_s))
        local image_target_rt_s = image.scale(image_target_rt, max_sz, 'bilinear')

	    for i_row = 1, image_target_rt_s:size()[2] - imageSize + 1, sampleStep do
	      for i_col = 1, image_target_rt_s:size()[3] - imageSize + 1, sampleStep do
	          image.save(folder_target_patch .. count .. '.png', image_target_rt_s[{{1, 3}, {i_row, i_row + imageSize - 1}, {i_col, i_col + imageSize - 1}}])
	          count = count + 1;
	      end
	    end
       
      end
    end
  end

end

  return flag_state
end

return {
  state = run_AG
}
