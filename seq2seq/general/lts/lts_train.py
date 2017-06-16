# -*- coding: utf-8 -*-
import sys
import os
import random
import re
import time
import torch 
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
from lts_model import *
from data_utils import *
import psutil

proc = psutil.Process(os.getpid())

def init_command_line(argv):
    from argparse import ArgumentParser
    usage = "seq2seq"
    description = ArgumentParser(usage)
    description.add_argument("--w2v_path", type=str, default="/users3/yfwang/rg_data/w2v/")
    description.add_argument("--corpus_path", type=str, default="/users3/yfwang/rg_data/corpus/")
    description.add_argument("--post_w2v", type=str, default="ecm_filt_tab_q_500e.w2v")
    description.add_argument("--reply_w2v", type=str, default="ecm_filt_tab_a_500e.w2v")
    description.add_argument("--post_file", type=str, default="ecm_filt_tab.q")
    description.add_argument("--reply_file", type=str, default="ecm_filt_tab.a")

    description.add_argument("--batch_size", type=int, default=64)
    description.add_argument("--hidden_size", type=int, default=1024)
    description.add_argument("--num_layers", type=int, default=1)
    description.add_argument("--max_length", type=int, default=10)
    description.add_argument("--lr", type=float, default=0.001)
    description.add_argument("--dropout", type=float, default=0.25)
    description.add_argument("--teach_forcing", type=int, default=1)

    description.add_argument("--epochs", type=int, default=10)
    description.add_argument("--shuffle", type=int, default=1)
    description.add_argument("--print_every", type=int, default=500, help="print every batches when training")
    description.add_argument("--save_model", type=int, default=1)
    description.add_argument("--weights", type=str, default=None)

    return description.parse_args(argv)

opts = init_command_line(sys.argv[1:])
print "Configure:"
print " post_w2v:",opts.w2v_path + opts.post_w2v
print " reply_w2v:",opts.w2v_path + opts.reply_w2v
print " post_file:",opts.corpus_path + opts.post_file
print " reply_file:",opts.corpus_path + opts.reply_file

print " batch_size:",opts.batch_size
print " hidden_size:",opts.hidden_size
print " num_layers:",opts.num_layers
print " max_length:",opts.max_length
print " learning rate:",opts.lr
print " dropout:",opts.dropout
print " teach_forcing:",opts.teach_forcing

print " epochs:",opts.epochs
print " shuffle:",opts.shuffle
print " print_every:",opts.print_every
print " save_model:",opts.save_model
print " weights:",opts.weights
print ""

'''单个batch的训练函数'''
def train_batch(post_tensor_batch,reply_tensor_batch,model,model_optimizer,criterion):
    loss = 0
    model_optimizer.zero_grad()

    list_pred = model(post_tensor_batch,reply_tensor_batch)

    # 预测的每个字的loss相加，构成整句的loss
    for idx,reply_tensor in enumerate(reply_tensor_batch):
        loss_s = criterion(list_pred[idx],Variable(reply_tensor).cuda())
        loss += loss_s
        #print "loss_s:",loss_s.data[0]

    clip_grad_norm(model.parameters, 5)
    loss.backward()
    model_optimizer.step()
    
    return loss.data[0]

# 多轮训练函数
def train_model(post_word2index,reply_word2index,post_index2word,reply_index2word,corpus_pairs,model,
                model_optimizer,criterion,epochs,batch_size,max_length,print_every,save_model,shuffle):
    print "start training..."
    model.train()
    for ei in range(epochs):
        print "Iteration {}: ".format(ei+1)
        epoch_loss = 0
        every_loss = 0
        t0 = time.time()
        pairs_batches,num_batches = buildingPairsBatch(corpus_pairs,batch_size,shuffle=shuffle)
        print "num_batches:",num_batches

        idx_batch = 0
        for post_tensor_batch,reply_tensor_batch in getTensorsPairsBatch(post_word2index,reply_word2index,pairs_batches):
            loss = train_batch(post_tensor_batch,reply_tensor_batch,model,model_optimizer,criterion)
            epoch_loss += loss
            every_loss += loss
            if (idx_batch+1)%print_every == 0:
                every_avg_loss = every_loss/(max_length*(idx_batch+1))
                #every_loss = 0
                print "{} batches finished, avg_loss:{}".format(idx_batch+1, every_avg_loss)
            idx_batch += 1

        if save_model:
            torch.save(model.state_dict(), "./seq2seq_parameters_IterEnd")
        
        print "memory percent: %.2f%%" % (proc.memory_percent())
        mem_info = proc.memory_info()
        res_mem_use = mem_info[0]
        print "res_mem_use: {:.2f}MB".format(float(res_mem_use)/1024/1024)

        epoch_avg_loss = epoch_loss/(max_length*num_batches)
        print "epoch_avg_loss:",epoch_avg_loss
        print "Iteration time:",time.time()-t0
        print "=============================================" 
        print ""

if __name__ == '__main__':
    unk_char = '<unk>'
    post_ctable,reply_ctable,corpus_pairs = readingData(opts.w2v_path+opts.post_w2v, opts.w2v_path+opts.reply_w2v, 
                                                        opts.corpus_path+opts.post_file, opts.corpus_path+opts.reply_file,
                                                        unk_char, opts.max_length)

    print " post dict size:",post_ctable.getDictSize()
    print " post emb size:",post_ctable.getEmbSize()
    print " reply dict size:",reply_ctable.getDictSize()
    print " reply emb size:",reply_ctable.getEmbSize()
    print ""

    seq2seq_model = Seq2Seq(post_ctable.getEmbMatrix(),reply_ctable.getEmbMatrix(),post_ctable.getDictSize(),
                            reply_ctable.getDictSize(),post_ctable.getEmbSize(),opts.hidden_size,opts.batch_size,
                            opts.num_layers,opts.dropout,opts.max_length,opts.teach_forcing).cuda()
    seq2seq_model.init_parameters()
    # 加载保存好的模型继续训练
    if opts.weights != None:
        print "load weights..."
        seq2seq_model.load_state_dict(torch.load(opts.weights))

    seq2seq_model.parameters = filter(lambda p: p.requires_grad, seq2seq_model.parameters())
    model_optimizer = optim.Adam(seq2seq_model.parameters, lr=opts.lr)
    criterion = nn.CrossEntropyLoss()
    
    print "memory percent: %.2f%%" % (proc.memory_percent())
    mem_info = proc.memory_info()
    res_mem_use = mem_info[0]
    print "res_mem_use: {:.2f}MB".format(float(res_mem_use)/1024/1024)
    print ""

    train_model(post_ctable.getWord2Index(),reply_ctable.getWord2Index(),post_ctable.getIndex2Word(),reply_ctable.getIndex2Word(),
                corpus_pairs,seq2seq_model,model_optimizer,criterion,opts.epochs,opts.batch_size,opts.max_length,opts.print_every,
                opts.save_model,opts.shuffle)
    print ""
