# -*- coding: utf-8 -*-
import time
import math
import torch 
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from memn2n import *

class Encoder(nn.Module):
    def __init__(self, emb_size, hidden_size):
        super(Encoder,self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size

        self.gru = nn.GRU(self.emb_size,self.hidden_size,batch_first=True)

    def forward(self, input, hidden):
        # input: batch_size*1*hidden_size
        # hidden: 1*batch_size*hidden_size
        output,hidden = self.gru(input,hidden)

        #return output,hidden
        return hidden

class Decoder(nn.Module):
    def __init__(self, decoder_dict_size, hidden_size, emb_size):
        super(Decoder, self).__init__()
        self.decoder_dict_size = decoder_dict_size
        self.hidden_size = hidden_size
        self.emb_size = emb_size

        # batch_first=True
        self.gru = nn.GRU(self.emb_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.decoder_dict_size)

    def forward(self, input, hidden):
        # input: batch_size*1*emb_size
        output, hidden = self.gru(input, hidden)

        output = self.out(output.squeeze())
        return output, hidden

class MemN2N_Seq2Seq(nn.Module):
    def __init__(self, post_dict_size, mem_dict_size, dec_dict_size, mem_size, emb_size, hidden_size, batch_size, 
                hops, max_length, dropout, teach_forcing, memn2n_type, **kw):
        super(MemN2N_Seq2Seq, self).__init__()
        self.post_dict_size = post_dict_size
        self.mem_dict_size = mem_dict_size
        self.dec_dict_size = dec_dict_size
        self.mem_size = mem_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.hops = hops
        self.max_length = max_length
        self.dropout = dropout
        self.teach_forcing = teach_forcing
        self.memn2n_type = memn2n_type
        self.using_w2v = False

        if self.memn2n_type == 0:
            print "Adjacent:"
            self.encoder = AjcnMemN2N(self.mem_size, self.hidden_size, self.emb_size, self.max_length, self.batch_size, self.hops)
        else:
            print "Layer-wise:"
            self.encoder = LayerMemN2N(self.mem_size, self.hidden_size, self.emb_size, self.max_length, self.batch_size, self.hops)
        self.decoder = Decoder(self.dec_dict_size, self.hidden_size, self.emb_size)

        # embedding层
        self.post_embedding = nn.Embedding(self.post_dict_size,self.emb_size)
        self.mem_embedding = nn.Embedding(self.mem_dict_size,self.emb_size)
        self.dec_embedding = nn.Embedding(self.dec_dict_size,self.emb_size)

        # 初始化embedding层参数
        if kw.has_key('post_emb_matrix') and kw.has_key('mem_emb_matrix') and kw.has_key('dec_emb_matrix'):
            print "get emb_matrix"
            self.post_embedding.weight = nn.Parameter(kw['post_emb_matrix'])
            self.mem_embedding.weight = nn.Parameter(kw['mem_emb_matrix'])
            self.dec_embedding.weight = nn.Parameter(kw['dec_emb_matrix'])
            self.using_w2v = True
            print ""

        # dropout层
        '''训练时调用Module.train(),将Dropout的self.training置为True
        预测时调用Module.eval(),将self.training置为False; 或者可以在预测时不使用dropout层'''
        self.dropout = nn.Dropout(self.dropout)

    ###########################################################
    # 初始化方法
    def init_parameters(self):
        print "using_word2vec:",self.using_w2v
        n = 0
        stdv = 1.0 / math.sqrt(self.hidden_size)
        #for weight in self.parameters():
        for name, weight in self.named_parameters():
            if weight.requires_grad:
                if "embedding" in name and self.using_w2v:
                    print "no need to initialize ",name
                    pass
                else:
                    #print name," ",weight.size()
                    if weight.ndimension() < 2:
                        #weight.data = init.uniform(weight.data,-stdv,stdv)
                        weight.data = torch.zeros(weight.size(0)).cuda()
                    else:
                        weight.data = self.xavier_uniform(weight.data)
                n += 1
        print "{} weights requires grad.".format(n)

    def calculate_fan_in_and_fan_out(self,tensor):
        if tensor.ndimension() < 2:
            raise ValueError("fan in and fan out can not be computed for tensor of size ", tensor.size())

        if tensor.ndimension() == 2:  # Linear
            fan_in = tensor.size(1)
            fan_out = tensor.size(0)
        else:
            num_input_fmaps = tensor.size(1)
            num_output_fmaps = tensor.size(0)
            receptive_field_size = 1
            if tensor.dim() > 2:
                receptive_field_size = tensor[0][0].numel()
            fan_in = num_input_fmaps * receptive_field_size
            fan_out = num_output_fmaps * receptive_field_size

        return fan_in, fan_out

    def xavier_uniform(self, tensor, gain=1):
        fan_in, fan_out = self.calculate_fan_in_and_fan_out(tensor)
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std

        return tensor.uniform_(-a, a)
    ###########################################################

    def forward(self, post_tensor_batch, reply_tensor_batch, mems_tensor_batch,ini_idx):
        '''post_tensor_batch: batch_size*max_length
        mems_tensor_batch为list, len=mem_size
        mems_tensor_batch[i]: batch_size*max_length'''
        list_decoder_output = []
        inputs = [None]*self.mem_size

        for i in range(self.mem_size):
            # input: batch_size*max_length*emb_size
            input = self.mem_embedding(Variable(mems_tensor_batch[i]).cuda())
            inputs[i] = self.dropout(input)

        # questions: batch_size*max_length*emb_size
        questions = self.post_embedding(Variable(post_tensor_batch).cuda())
        questions = self.dropout(questions)

        # encoder_output: batch_size*1*hidden_size
        encoder_output = self.encoder(inputs,questions)

        decoder_input = torch.LongTensor([ini_idx]*self.batch_size).cuda()

        decoder_hidden = encoder_output.view(1,self.batch_size,-1)

        # reply_tensor_batch: batch_size*max_length
        for reply_tensor in reply_tensor_batch.t():
            decoder_embedded = self.dec_embedding(Variable(decoder_input).unsqueeze(1))
            decoder_output,decoder_hidden = self.decoder(decoder_embedded, decoder_hidden)
            list_decoder_output.append(decoder_output)
            if self.teach_forcing:
                decoder_input = reply_tensor.cuda()
            else:
                top_values,top_indices = decoder_output.data.topk(1)
                decoder_input = top_indices.squeeze(1)

        return list_decoder_output

    def predict(self, post_tensor_batch, mems_tensor_batch, reply_index2word,ini_idx,sep_char=''):
        predict_words_array = [['' for i in range(self.max_length)] for i in range(self.batch_size)]
        predict_sentences = ["" for i in range(self.batch_size)]

        inputs = [None]*self.mem_size

        for i in range(self.mem_size):
            # input: batch_size*max_length*emb_size
            input = self.mem_embedding(Variable(mems_tensor_batch[i]).cuda())
            inputs[i] = input

        questions = self.post_embedding(Variable(post_tensor_batch).cuda())

        encoder_output = self.encoder(inputs,questions)

        decoder_input = torch.LongTensor([ini_idx]*self.batch_size).cuda()

        decoder_hidden = encoder_output.view(1,self.batch_size,-1)

        for di in range(self.max_length):
            decoder_embedded = self.dec_embedding(Variable(decoder_input).unsqueeze(1))

            decoder_output,decoder_hidden = self.decoder(decoder_embedded, decoder_hidden)
            top_values,top_indices = decoder_output.data.topk(1)
            batch_topi = [top_indices[i][0] for i in range(self.batch_size)]
            for i in range(self.batch_size):
                predict_words_array[i][di] = reply_index2word[batch_topi[i]]
                #predict_words_array[i][di] = str(batch_topi[i])
            decoder_input = top_indices.squeeze(1)
            
        # 预测的句子以sep_char分隔
        for i in range(self.batch_size):
            predict_sentences[i] = sep_char.join(predict_words_array[i])
        return predict_sentences