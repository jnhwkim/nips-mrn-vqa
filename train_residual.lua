------------------------------------------------------------------------------
--  Multimodal Residual Networks for Visual QA
--  Jin-Hwa Kim, Sang-Woo Lee, Dong-Hyun Kwak, Min-Oh Heo, 
--    Jeonghee Kim, Jung-Woo Ha, Byoung-Tak Zhang 
--  https://arxiv.org/abs/1606.01455
--
--  This code is based on 
--    https://github.com/VT-vision-lab/VQA_LSTM_CNN/blob/master/train.lua
-----------------------------------------------------------------------------

require 'nn'
require 'rnn'
require 'torch'
require 'optim'
require 'cutorch'
require 'cunn'
require 'hdf5'
cjson=require('cjson') 

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Visual Question Answering model')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-input_img_h5','data_train-val_test-dev_2k/data_res.h5','path to the h5file containing the image feature')
cmd:option('-input_ques_h5','data_train-val_test-dev_2k/data_prepro.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','data_train-val_test-dev_2k/data_prepro.json','path to the json file containing additional info and vocab')
cmd:option('-input_skip','skipthoughts_model','path to skipthoughts_params')

-- Model parameter settings
cmd:option('-learning_rate',3e-4,'learning rate for rmsprop')
cmd:option('-learning_rate_decay_start', -1, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 50000, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-batch_size',200,'batch_size for each iterations')
cmd:option('-max_iters', 250000, 'max number of iterations to run for ')
cmd:option('-input_encoding_size', 620, 'he encoding size of each token in the vocabulary')
cmd:option('-rnn_size',2400,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-common_embedding_size', 1200, 'size of the common embedding vector')
cmd:option('-num_output', 2000, 'number of output answers')
cmd:option('-img_norm', 1, 'normalize the image feature. 1 = normalize, 0 = not normalize')
cmd:option('-model_name', 'MRN', 'model name')
cmd:option('-label','','model label')
cmd:option('-num_layers', 3, '# of layers of Multimodal Residual Networks')
cmd:option('-dropout', .5, 'dropout probability for joint functions')

--check point
cmd:option('-save_checkpoint_every', 25000, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', 'model/', 'folder to save checkpoints')
cmd:option('-load_checkpoint_path', '', 'path to saved checkpoint')
cmd:option('-previous_iters', 0, 'previous # of iterations to get previous learning rate')
cmd:option('-kick_interval', 50000, 'interval of kicking the learning rate as its double')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-seed', 1232, 'random number generator seed to use')

opt = cmd:parse(arg)
print(opt)

torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

require 'misc.RNNUtils'
if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1)
end

------------------------------------------------------------------------
-- Setting the parameters
------------------------------------------------------------------------
local model_name = opt.model_name..opt.label..'_L'..opt.num_layers
local num_layers = opt.num_layers
local model_path = opt.checkpoint_path
local batch_size=opt.batch_size
local embedding_size_q=opt.input_encoding_size
local rnn_size_q=opt.rnn_size
local common_embedding_size=opt.common_embedding_size
local noutput=opt.num_output
local dropout=opt.dropout
local decay_factor = 0.99997592083 -- 50000
local question_max_length=26
paths.mkdir(model_path)

------------------------------------------------------------------------
-- Loading Dataset
------------------------------------------------------------------------
local file = io.open(opt.input_json, 'r')
local text = file:read()
file:close()
json_file = cjson.decode(text)

print('DataLoader loading h5 file: ', opt.input_ques_h5)
dataset = {}
local h5_file = hdf5.open(opt.input_ques_h5, 'r')

dataset['question'] = h5_file:read('/ques_train'):all()
dataset['lengths_q'] = h5_file:read('/ques_length_train'):all()
dataset['img_list'] = h5_file:read('/img_pos_train'):all()
dataset['answers'] = h5_file:read('/answers'):all()
h5_file:close()

print('DataLoader loading h5 file: ', opt.input_img_h5)
local h5_file = hdf5.open(opt.input_img_h5, 'r')
dataset['fv_im'] = h5_file:read('/images_train'):all()
h5_file:close()
local nhimage=dataset['fv_im']:size(2)
print('nhimage', nhimage)

dataset['question'] = right_align(dataset['question'],dataset['lengths_q'])

-- Normalize the image feature
if opt.img_norm == 1 then
   local nm=torch.sqrt(torch.sum(torch.cmul(dataset['fv_im'],dataset['fv_im']),2)) 
   dataset['fv_im']=torch.cdiv(dataset['fv_im'],torch.repeatTensor(nm,1,nhimage)):float() 
end

local count = 0
for i, w in pairs(json_file['ix_to_word']) do count = count + 1 end
local vocabulary_size_q=count

collectgarbage() 

------------------------------------------------------------------------
--Design Parameters and Network Definitions
------------------------------------------------------------------------
print('Building the model...')

buffer_size_q=dataset['question']:size()[2]

-- Skip-Thought Vectors (Kiros et al., 2015)
-- lookup = nn.LookupTableMaskZero(vocabulary_size_q, embedding_size_q)
if opt.num_output == 1000 then lookupfile = 'lookup_fix.t7'
elseif opt.num_output == 2000 then lookupfile = 'lookup_2k.t7' 
elseif opt.num_output == 3000 then lookupfile = 'lookup_3k.t7' 
end
lookup = torch.load(paths.concat(opt.input_skip, lookupfile))
assert(lookup.weight:size(1)==vocabulary_size_q+1)  -- +1 for zero
assert(lookup.weight:size(2)==opt.input_encoding_size)
gru = torch.load(paths.concat(opt.input_skip, 'gru.t7'))
-- Bayesian GRUs
bgru = nn.GRU(embedding_size_q, rnn_size_q, false, .25, true)  -- Cho et al. (2014); Gal & Ghahramani (2016)
skip_params = gru:parameters()
bgru:migrate(skip_params)
bgru:trimZero(1)  -- Kim et al. (2016a) https://github.com/Element-Research/rnn#rnn.TrimZero
gru = nil
collectgarbage()

--embedding: word-embedding
embedding_net_q=nn.Sequential()
            :add(lookup)
            :add(nn.SplitTable(2))

--encoder: RNN body
encoder_net_q=nn.Sequential()
            :add(nn.Sequencer(bgru))
            :add(nn.SelectTable(question_max_length))

require 'netdef.MRN'
multimodal_net=netdef[opt.model_name](rnn_size_q,nhimage,common_embedding_size,dropout,num_layers,noutput)
print(multimodal_net)

--criterion
criterion=nn.CrossEntropyCriterion()

if opt.gpuid >= 0 then
   print('shipped data function to cuda...')
   embedding_net_q = embedding_net_q:cuda()
   encoder_net_q = encoder_net_q:cuda()
   multimodal_net = multimodal_net:cuda()
   criterion = criterion:cuda()
end

--Processings
embedding_w_q,embedding_dw_q=embedding_net_q:getParameters() 
encoder_w_q,encoder_dw_q=encoder_net_q:getParameters() 
multimodal_w,multimodal_dw=multimodal_net:getParameters()

if paths.filep(opt.load_checkpoint_path) then
   print('loading checkpoint model...')
   -- loading the model
   model_param=torch.load(opt.load_checkpoint_path);
   if embedding_w_q:size(1) ~= model_param['embedding_w_q']:size(1) then
      print('warning: `embedding_w_q` size does not match!')
   end
   -- trying to use the precedding parameters
   embedding_w_q:copy(model_param['embedding_w_q']:resizeAs(embedding_w_q))
   encoder_w_q:copy(model_param['encoder_w_q']);
   multimodal_w:copy(model_param['multimodal_w']);
else
   multimodal_w:uniform(-0.08, 0.08) 
end

sizes={encoder_w_q:size(1),embedding_w_q:size(1),multimodal_w:size(1)} 

-- optimization parameter
local optimize={} 
optimize.maxIter=opt.max_iters 
optimize.learningRate=opt.learning_rate
optimize.update_grad_per_n_batches=1

optimize.winit=join_vector({encoder_w_q,embedding_w_q,multimodal_w}) 
print('nParams=',optimize.winit:size(1))

------------------------------------------------------------------------
-- Next batch for train
------------------------------------------------------------------------
function dataset:next_batch()
   local qinds=torch.LongTensor(batch_size):fill(0) 
   local iminds=torch.LongTensor(batch_size):fill(0)  
   local nqs=dataset['question']:size(1) 
   -- we use the last val_num data for validation (the data already randomlized when created)
   for i=1,batch_size do
      qinds[i]=torch.random(nqs) 
      iminds[i]=dataset['img_list'][qinds[i]] 
   end

   local fv_sorted_q=dataset['question']:index(1,qinds) 
   local fv_im=dataset['fv_im']:index(1,iminds) 
   local labels=dataset['answers']:index(1,qinds) 
   
   -- ship to gpu
   if opt.gpuid >= 0 then
      fv_sorted_q=fv_sorted_q:cuda() 
      fv_im = fv_im:cuda()
      labels = labels:cuda()
   end
   return fv_sorted_q,fv_im,labels,batch_size 
end

------------------------------------------------------------------------
-- Objective Function and Optimization
------------------------------------------------------------------------
function JdJ(x)
   local params=split_vector(x,sizes) 
   --load x to net parameters--
   if encoder_w_q~=params[1] then
      encoder_w_q:copy(params[1]) 
   end
   if embedding_w_q~=params[2] then
      embedding_w_q:copy(params[2]) 
   end
   if multimodal_w~=params[3] then
      multimodal_w:copy(params[3]) 
   end

   --clear gradients--
   encoder_dw_q:zero()
   embedding_dw_q:zero() 
   multimodal_dw:zero() 

   --grab a batch--
   local fv_sorted_q,fv_im,labels,batch_size=dataset:next_batch() 

   local model = nn.Sequential()
      :add(nn.ParallelTable()
         :add(nn.Sequential()
            :add(embedding_net_q)
            :add(encoder_net_q))
         :add(nn.Identity()))
      :add(multimodal_net)

   local scores = model:forward({fv_sorted_q, fv_im})
   local f=criterion:forward(scores,labels)
   local dscores=criterion:backward(scores,labels)
   model:backward(fv_sorted_q, dscores)
      
   gradients=join_vector({encoder_dw_q,embedding_dw_q,multimodal_dw})
   gradients:clamp(-10,10) 
   if running_avg == nil then
      running_avg = f
   end
   running_avg=running_avg*0.95+f*0.05
   return f,gradients 
end

------------------------------------------------------------------------
-- Training
------------------------------------------------------------------------
local state={}
optimize.learningRate=optimize.learningRate*decay_factor^opt.previous_iters
optimize.learningRate=optimize.learningRate*2^math.min(2, math.floor(opt.previous_iters/opt.kick_interval))
for iter = opt.previous_iters + 1, opt.max_iters do
   if iter%opt.save_checkpoint_every == 0 then
      paths.mkdir(model_path..'save')
      torch.save(string.format(model_path..'save/'..model_name..'_iter%d.t7',iter),
         {encoder_w_q=encoder_w_q,embedding_w_q=embedding_w_q,multimodal_w=multimodal_w}) 
   end
   if iter%100 == 0 then
      print('training loss: ' .. running_avg, 'on iter: ' .. iter .. '/' .. opt.max_iters)
   end
   -- double learning rate at two iteration points
   if iter==opt.kick_interval or iter==opt.kick_interval*2 then
      optimize.learningRate=optimize.learningRate*2
      print('learining rate:', optimize.learningRate)
   end
   if opt.previous_iters == iter-1 then
      print('learining rate:', optimize.learningRate)
   end
   optim.rmsprop(JdJ, optimize.winit, optimize, state)
   
   optimize.learningRate=optimize.learningRate*decay_factor 
   if iter%5 == 0 then -- change this to smaller value if out of the memory
      collectgarbage()
   end
end

-- Saving the final model
torch.save(string.format(model_path..model_name..'.t7',i),
   {encoder_w_q=encoder_w_q,embedding_w_q=embedding_w_q,multimodal_w=multimodal_w}) 
