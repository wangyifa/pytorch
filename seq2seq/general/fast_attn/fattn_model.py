# -*- coding: utf-8 -*-
import sys
import math
import time
import torch 
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class Encoder(nn.Module):
    def __init__(self, emb_size, hidden_size, num_layers, dropout):
        super(Encoder,self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.gru = nn.GRU(self.emb_size,self.hidden_size,num_layers=self.num_layers,batch_first=True,dropout=self.dropout)

    def forward(self, input, hidden):
        # input: batch_size*1*hidden_size
        # hidden: num_layers*batch_size*hidden_size
        output,hidden = self.gru(input,hidden)

        return output,hidden

class AttnDecoder(nn.Module):
    def __init__(self, decoder_dict_size, hidden_size, emb_size, num_layers, dropout):
        super(AttnDecoder, self).__init__()
        self.decoder_dict_size = decoder_dict_size
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.gru = nn.GRU(self.hidden_size+self.emb_size,self.hidden_size,num_layers=self.num_layers,batch_first=True,dropout=self.dropout)
        self.out = nn.Linear(self.hidden_size,self.decoder_dict_size)

    def forward(self, input, hidden, encoder_outputs):
        '''hidden: num_layers*batch_size*hidden_size
        encoder_outputs: batch_size*max_length*hidden_size
        e_j = tanh(h_t-1^t*eo_j)
        attn_weights: batch_size*maxlength'''
        weight = torch.bmm(encoder_outputs,hidden[self.num_layers-1].unsqueeze(2)).squeeze(2)
        #print weight
        attn_weights = F.softmax(weight)

        '''encoder_outputs: batch_size*max_length*hidden_size
        attn_weights和encoder_outputs相乘 即max_length个输出乘权重后相加 c=∑(αj*hj)'''
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)

        # output_combine: batch_size*1*hidden_size+emb_size
        output_combine = torch.cat((input, attn_applied), 2)
        output, hidden = self.gru(output_combine, hidden)

        output = self.out(output.squeeze(1))

        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder_emb_matrix, decoder_emb_matrix, encoder_dict_size, decoder_dict_size, 
                emb_size, hidden_size, batch_size, num_layers, dropout, max_length, teach_forcing):
        super(Seq2Seq, self).__init__()
        self.encoder_dict_size = encoder_dict_size
        self.decoder_dict_size = decoder_dict_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_length = max_length
        self.teach_forcing = teach_forcing

        self.encoder = Encoder(self.emb_size, self.hidden_size, self.num_layers, self.dropout)
        self.decoder = AttnDecoder(self.decoder_dict_size, self.hidden_size, self.emb_size, self.num_layers, self.dropout)

        # embedding层
        self.enc_embedding = nn.Embedding(self.encoder_dict_size,self.emb_size)
        self.enc_embedding.weight = nn.Parameter(encoder_emb_matrix)
        self.dec_embedding = nn.Embedding(self.decoder_dict_size,self.emb_size)
        self.dec_embedding.weight = nn.Parameter(decoder_emb_matrix)
        
        '''dropout层: 训练时调用Module.train(),将Dropout的self.training置为True
        预测时调用Module.eval(),将self.training置为False; 或者可以在预测时不使用dropout层'''
        self.dropout = nn.Dropout(self.dropout)

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

    # 初始化方法
    def init_parameters(self):
        n = 0
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for name, weight in self.named_parameters():
            if weight.requires_grad:
                if name == "dec_embedding.weight" or name == "enc_embedding.weight":
                    print "no need to initialize",name
                    pass
                else:
                    if weight.ndimension() < 2:
                        #weight.data = init.uniform(weight.data, -stdv, stdv)
                        weight.data = torch.zeros(weight.size(0)).cuda()
                    else:
                        weight.data = self.xavier_uniform(weight.data)
                n += 1
        print "{} weights requires grad.".format(n)

    def forward(self, post_tensor_batch, reply_tensor_batch, ini_idx):
        encoder_hidden = Variable(torch.zeros(self.num_layers,self.batch_size,self.hidden_size)).cuda()
        list_encoder_output = []
        list_decoder_output = []

        # 循环次数为max_length
        for post_tensor in post_tensor_batch:
            encoder_embedded = self.enc_embedding(Variable(post_tensor).cuda()).unsqueeze(1)
            encoder_output,encoder_hidden = self.encoder(encoder_embedded,encoder_hidden)
            list_encoder_output.append(self.dropout(encoder_output))

        # max_length个Tensor(batch_size*1*hidden_size) -> batch_size*max_length*hidden_size
        encoder_outputs = torch.cat(tuple(list_encoder_output),1)

        decoder_input = torch.LongTensor([ini_idx]*self.batch_size).cuda()
        # dropout层只作用于RNN层之间的值传递
        decoder_hidden = self.dropout(encoder_hidden)

        for reply_tensor in reply_tensor_batch:
            decoder_embedded = self.dec_embedding(Variable(decoder_input)).unsqueeze(1)
            decoder_output,decoder_hidden = self.decoder(decoder_embedded, decoder_hidden, encoder_outputs)
            
            list_decoder_output.append(decoder_output)
            if self.teach_forcing:
                # reply_variable：batch_size *
                decoder_input = reply_tensor.cuda()
            else:
                top_values,top_indices = decoder_output.data.topk(1)
                decoder_input = top_indices.squeeze(1)

        return list_decoder_output

    # 预测用函数，输出一个batch对应的预测结果的list
    def predict(self, post_tensor_batch, reply_index2word, ini_idx, sep_char=''):
        encoder_hidden = Variable(torch.zeros(self.num_layers,self.batch_size,self.hidden_size)).cuda()

        list_encoder_output = []
        predict_words_array = [['' for i in range(self.max_length)] for i in range(self.batch_size)]
        predict_sentences = ["" for i in range(self.batch_size)]

        for post_tensor in post_tensor_batch:
            encoder_embedded = self.enc_embedding(Variable(post_tensor).cuda()).unsqueeze(1)
            encoder_output,encoder_hidden = self.encoder(encoder_embedded,encoder_hidden)
            list_encoder_output.append(encoder_output)
        
        encoder_outputs = torch.cat(tuple(list_encoder_output),1)

        decoder_input = torch.LongTensor([ini_idx]*self.batch_size).cuda()
        decoder_hidden = encoder_hidden

        for di in range(self.max_length):
            decoder_embedded = self.dec_embedding(Variable(decoder_input)).unsqueeze(1)
            decoder_output,decoder_hidden = self.decoder(decoder_embedded, decoder_hidden, encoder_outputs)
            top_values,top_indices = decoder_output.data.topk(1)
            batch_topi = [top_indices[i][0] for i in range(self.batch_size)]
            for i in range(self.batch_size):
                predict_words_array[i][di] = reply_index2word[batch_topi[i]]
            decoder_input = top_indices.squeeze(1)

        # 预测的句子以sep_char分隔
        for i in range(self.batch_size):
            predict_sentences[i] = sep_char.join(predict_words_array[i])
        return predict_sentences
