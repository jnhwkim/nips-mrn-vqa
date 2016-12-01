netdef = {}

--Model (b) in Figure 3
function netdef.AxB2(nhA,nhB,nhcommon,dropout)
   dropout = dropout or 0 
   local AxB = nn.Sequential()
      :add(nn.ParallelTable()
         :add(nn.Sequential()
            :add(nn.Dropout(dropout))
            :add(nn.Linear(nhA,nhcommon))
            :add(nn.Tanh()))
         :add(nn.Sequential()
            :add(nn.Dropout(dropout))
            :add(nn.Linear(nhB,nhB))
            :add(nn.Tanh())
            :add(nn.Dropout(dropout))
            :add(nn.Linear(nhB,nhcommon))
            :add(nn.Tanh())))
      :add(nn.CMulTable())
   return AxB
end

function netdef.AxB(nhA,nhB,nhcommon,dropout)
   dropout = dropout or 0 
   local AxB = nn.Sequential()
      :add(nn.ParallelTable()
         :add(nn.Sequential()
            :add(nn.Dropout(dropout))
            :add(nn.Linear(nhA,nhcommon))
            :add(nn.Tanh()))
         :add(nn.Sequential()
            :add(nn.Dropout(dropout))
            :add(nn.Linear(nhB,nhcommon))
            :add(nn.Tanh())))
      :add(nn.CMulTable())
   return AxB
end

--Multimodal Residual Learning
--multimodal way of combining different inputs using residual learning on question
function netdef.MRN(rnn_size_q,nhimage,common_embedding_size,joint_dropout,num_layers,noutput)
   local dropout = .5
   if num_layers == 1 then
      multimodal_net=nn.Sequential()
            :add(nn.ConcatTable()  -- layer 1
               :add(netdef.AxB2(rnn_size_q,nhimage,common_embedding_size,joint_dropout))
               :add(nn.Sequential()
                  :add(nn.SelectTable(1)):add(nn.Dropout(dropout))
                  :add(nn.Linear(rnn_size_q, common_embedding_size))))
            :add(nn.CAddTable())
            :add(nn.Dropout(dropout))
            :add(nn.Linear(common_embedding_size,noutput))
      return multimodal_net
   end
   multimodal_net=nn.Sequential()
         :add(nn.ConcatTable()  -- layer 1
            :add(netdef.AxB2(rnn_size_q,nhimage,common_embedding_size,joint_dropout))
            :add(nn.Sequential()
               :add(nn.SelectTable(1)):add(nn.Dropout(dropout))
               :add(nn.Linear(rnn_size_q, common_embedding_size)))
            :add(nn.SelectTable(2)))
         :add(nn.ConcatTable()
            :add(nn.Sequential()
               :add(nn.NarrowTable(1,2)):add(nn.CAddTable()))
            :add(nn.SelectTable(3)))
   for i=3,num_layers do multimodal_net  -- option starts with 2
         :add(nn.ConcatTable()  -- middle layer
            :add(netdef.AxB2(common_embedding_size,nhimage,common_embedding_size,joint_dropout))
            :add(nn.Sequential()
               :add(nn.SelectTable(1)):add(nn.Dropout(dropout))
               :add(nn.Linear(common_embedding_size, common_embedding_size)))
            :add(nn.SelectTable(2)))
         :add(nn.ConcatTable()
            :add(nn.Sequential()
               :add(nn.NarrowTable(1,2)):add(nn.CAddTable()))
            :add(nn.SelectTable(3))) end multimodal_net
         :add(nn.ConcatTable()  -- last layer
            :add(netdef.AxB2(common_embedding_size,nhimage,common_embedding_size,joint_dropout))
            :add(nn.Sequential()
               :add(nn.SelectTable(1)):add(nn.Dropout(dropout))
               :add(nn.Linear(common_embedding_size, common_embedding_size))))
         :add(nn.CAddTable())
         :add(nn.Dropout(dropout))
         :add(nn.Linear(common_embedding_size,noutput))
   return multimodal_net
end

--Multimodal Non-Residual Learning
function netdef.MN(rnn_size_q,nhimage,common_embedding_size,dropout,num_layers,noutput)
   multimodal_net=nn.Sequential()
         :add(nn.ConcatTable()  -- layer 1
            :add(netdef.AxB2(rnn_size_q,nhimage,common_embedding_size,dropout))
            :add(nn.SelectTable(2)))
   for i=2,num_layers do multimodal_net
         :add(nn.ConcatTable()  -- middle layer
            :add(netdef.AxB2(common_embedding_size,nhimage,common_embedding_size,dropout))
            :add(nn.SelectTable(2))) end multimodal_net
         :add(nn.SelectTable(1))
         :add(nn.Dropout(dropout))
         :add(nn.Linear(common_embedding_size,noutput))
   return multimodal_net
end
