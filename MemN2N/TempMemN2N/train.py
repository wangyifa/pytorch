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
from seq2seq_model import MemN2N_Seq2Seq
from data_utils import *
import psutil

proc = psutil.Process(os.getpid())

def init_command_line(argv):
    from argparse import ArgumentParser
    usage = "memn2n"
    description = ArgumentParser(usage)
    description.add_argument("--w2v_path", type=str, default="/users3/yfwang/rg_data/w2v/")
    description.add_argument("--corpus_path", type=str, default="/users3/yfwang/rg_data/corpus/")
    description.add_argument("--corpus_w2v", type=str, default="jx1log_tab_all.w2v")
    description.add_argument("--post_file", type=str, default="jx1log_tab_train.q")
    description.add_argument("--reply_file", type=str, default="jx1log_tab_train.a")
    description.add_argument("--mem_file", type=str, default="jx1log_all_tab_train.mem")
    
    description.add_argument("--memn2n_type", type=int)
    description.add_argument("--mem_size", type=int, default=4)
    description.add_argument("--batch_size", type=int, default=32)
    description.add_argument("--hidden_size", type=int, default=1024)
    description.add_argument("--max_length", type=int, default=10)
    description.add_argument("--hops", type=int, default=3)

    description.add_argument("--epochs", type=int, default=10)
    description.add_argument("--lr", type=float, default=0.001)
    description.add_argument("--dropout", type=float, default=0.25)
    description.add_argument("--teach_forcing", type=int, default=1)
    description.add_argument("--using_w2v", type=int, default=1)
    description.add_argument("--shuffle", type=int, default=1)
    description.add_argument("--print_every", type=int, default=100, help="print every batches when training")
    description.add_argument("--save_model", type=int, default=1)

    description.add_argument("--weights", type=str, default=None)
    return description.parse_args(argv)

opts = init_command_line(sys.argv[1:])
print "Configure:"
print " corpus_w2v:",opts.w2v_path + opts.corpus_w2v
print " post_file:",opts.corpus_path + opts.post_file
print " reply_file:",opts.corpus_path + opts.reply_file
print " mem_file:",opts.corpus_path + opts.mem_file

if opts.memn2n_type != None:
    print " memn2n_type:",opts.memn2n_type
else:
    print "what's the type of memn2n model?"
    exit()
print " mem_size:",opts.mem_size
print " batch_size:",opts.batch_size
print " hidden_size:",opts.hidden_size
print " max_length:",opts.max_length
print " hops:",opts.hops

print " epochs:",opts.epochs
print " learning rate:",opts.lr
print " dropout:",opts.dropout
print " teach_forcing:",opts.teach_forcing
print " using_w2v:",opts.using_w2v
print " shuffle:",opts.shuffle
print " print_every:",opts.print_every
print " save_model:",opts.save_model

print " weights:",opts.weights
print ""

'''单个batch的训练函数'''
def train_batch(post_vari_batch,reply_vari_batch,mems_vari_batch,model,model_optimizer,criterion,ini_idx):
    loss = 0
    model_optimizer.zero_grad()

    list_pred = model(post_vari_batch,reply_vari_batch,mems_vari_batch,ini_idx)

    # 预测的每个字的loss相加，构成整句的loss
    for idx,reply_tensor in enumerate(reply_vari_batch.t()):
        loss_s = criterion(list_pred[idx],Variable(reply_tensor).cuda())
        loss += loss_s

    clip_grad_norm(model.parameters, 5)
    loss.backward()
    model_optimizer.step()
    
    return loss.data[0]

# 多轮训练函数
def train_model(corpus_word2index,ini_idx,corpus_pairs,model,model_optimizer,criterion,epochs,
                batch_size,max_length,mem_size,print_every,save_model,shuffle):
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
        for post_vari_batch,reply_vari_batch,mems_vari_batch in getTensorsPairsBatch(corpus_word2index,pairs_batches,mem_size):
            loss = train_batch(post_vari_batch,reply_vari_batch,mems_vari_batch,model,model_optimizer,criterion,ini_idx)
            epoch_loss += loss
            every_loss += loss
            if (idx_batch+1)%print_every == 0:
                every_avg_loss = every_loss/(max_length*(idx_batch+1))
                #every_loss = 0
                print "{} batches finished, avg_loss:{}".format(idx_batch+1, every_avg_loss)
            idx_batch += 1

        if save_model:
            torch.save(model.state_dict(), "./memn2n_parameters_IterEnd")
        
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
    ini_char = '</i>'
    unk_char = '<unk>'
    corpus_ctable,corpus_pairs = readingData(opts.w2v_path+opts.corpus_w2v, opts.corpus_path+opts.post_file, 
                                                                    opts.corpus_path+opts.reply_file, opts.corpus_path+opts.mem_file,
                                                                    ini_char, unk_char, opts.mem_size, opts.max_length)

    print " corpus dict size:",corpus_ctable.getDictSize()
    print " corpus emb size:",corpus_ctable.getEmbSize()
    print ""

    if opts.using_w2v:
        memn2n_seq2seq = MemN2N_Seq2Seq(corpus_ctable.getDictSize(),opts.mem_size,corpus_ctable.getEmbSize(),opts.hidden_size,
                                        opts.batch_size,opts.hops,opts.max_length,opts.dropout,opts.teach_forcing,
                                        opts.memn2n_type,emb_matrix=corpus_ctable.getEmbMatrix()).cuda()
    else:
        memn2n_seq2seq = MemN2N_Seq2Seq(corpus_ctable.getDictSize(),opts.mem_size,corpus_ctable.getEmbSize(),opts.hidden_size,
                                        opts.batch_size,opts.hops,opts.max_length,opts.dropout,opts.teach_forcing).cuda()
    memn2n_seq2seq.init_parameters()
    # 加载保存好的模型继续训练
    if opts.weights != None:
        print "load weights..."
        memn2n_seq2seq.load_state_dict(torch.load(opts.weights))

    # 去掉一些不需要更新的参数
    memn2n_seq2seq.parameters = filter(lambda p: p.requires_grad, memn2n_seq2seq.parameters())
    model_optimizer = optim.Adam(memn2n_seq2seq.parameters, lr=opts.lr)
    criterion = nn.CrossEntropyLoss()
    
    print "memory percent: %.2f%%" % (proc.memory_percent())
    mem_info = proc.memory_info()
    res_mem_use = mem_info[0]
    print "res_mem_use: {:.2f}MB".format(float(res_mem_use)/1024/1024)
    print ""

    corpus_word2index = corpus_ctable.getWord2Index()
    ini_idx = corpus_word2index[ini_char]
    train_model(corpus_word2index,ini_idx,corpus_pairs,memn2n_seq2seq,model_optimizer,criterion,opts.epochs,
                opts.batch_size,opts.max_length,opts.mem_size,opts.print_every,opts.save_model,opts.shuffle)
    print ""
