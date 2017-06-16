 #-*- coding: utf-8 -*-
import sys
import random
import re
import time
import torch 
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from seq2seq_model import MemN2N_Seq2Seq
from data_utils import *

def init_command_line(argv):
    from argparse import ArgumentParser
    usage = "memn2n"
    description = ArgumentParser(usage)
    description.add_argument("--w2v_path", type=str, default="/users3/yfwang/rg_data/w2v/")
    description.add_argument("--post_w2v", type=str, default="ecm_filt_tab_q_200e.w2v")
    description.add_argument("--reply_w2v", type=str, default="ecm_filt_tab_a_200e.w2v")
    description.add_argument("--mem_w2v", type=str, default="ecm_filt_tab_all_200e.w2v")
    
    description.add_argument("--memn2n_type", type=int, default=0)
    description.add_argument("--mem_size", type=int, default=2)
    description.add_argument("--batch_size", type=int, default=64)
    description.add_argument("--hidden_size", type=int, default=1024)
    description.add_argument("--max_length", type=int, default=10)
    description.add_argument("--hops", type=int, default=3)

    description.add_argument("--dropout", type=float, default=0.25)
    description.add_argument("--teach_forcing", type=int, default=1)
    description.add_argument("--print_every", type=int, default=500, help="print every batches when training")

    description.add_argument("--weights", type=str, default=None)
    return description.parse_args(argv)

opts = init_command_line(sys.argv[1:])
print "Configure:"
print " post_w2v:",opts.w2v_path + opts.post_w2v
print " reply_w2v:",opts.w2v_path + opts.reply_w2v
print " mem_w2v:",opts.w2v_path + opts.mem_w2v

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

print " dropout:",opts.dropout
print " teach_forcing:",opts.teach_forcing
print " print_every:",opts.print_every

print " weights:",opts.weights
print ""
    

# 读入一个句子的list，构建batch后进行预测
def predict_Sentences(post_word2index,mem_word2index,reply_index2word,unk_char,ini_char,ini_idx,model,
                        post_file_path,mem_file_path,print_every,batch_size,max_length,mem_size):
    model.eval()
    post_file = open(post_file_path,'r')
    mem_file = open(mem_file_path,'r') 
    post_sentences = []
    mem_sentences = []
    for line in post_file:
        line = line.strip('\n').strip()
        post_sentences.append(line)
    for line in mem_file:
        line = line.strip('\n').strip()
        mem_sentences.append(line)
    post_file.close()
    mem_file.close()

    list_pad_post = []
    for sentence in post_sentences:
        sentence = filteringSenten(post_word2index,sentence,unk_char,ini_char)
        sentence = paddingSenten(sentence,max_length)
        list_pad_post.append(sentence)

    tmp_mem = []
    list_pad_mem = []
    for memidx,sentence in enumerate(mem_sentences):
        sentence = filteringSenten(mem_word2index,sentence,unk_char,ini_char)
        sentence = paddingSenten(sentence,max_length)
        tmp_mem.append(sentence)
        if (memidx+1)%mem_size == 0:
            list_pad_mem.append(tmp_mem)
            tmp_mem = []
    
    #构造batch的list
    num_post = len(list_pad_post)
    post_batches = buildingSentenBatch(list_pad_post,batch_size)
    mem_batches = buildingMemBatch(list_pad_mem,batch_size)
    if len(post_batches) == len(mem_batches):
        print "num of batch:",len(post_batches)
        print ""

    predict_sentences = []
    idx_batch = 0
    for post_tensor_batch,mems_tensor_batch in getTensorsSentenMemBatch(post_word2index,mem_word2index,post_batches,mem_batches,mem_size):
        predict_batch = model.predict(post_tensor_batch,mems_tensor_batch,reply_index2word,ini_idx,sep_char=' ')
        predict_sentences.extend(predict_batch)
        if (idx_batch+1)%print_every == 0:
            print "{} batches finished".format(idx_batch+1)
        idx_batch += 1

    predict_sentences = predict_sentences[0:num_post]
    return post_sentences,list_pad_mem,predict_sentences

if __name__ == '__main__':
    ini_char = '</i>'
    unk_char = '<unk>'
    post_ctable,reply_ctable,mem_ctable = readingW2v(opts.w2v_path+opts.post_w2v,opts.w2v_path+opts.reply_w2v,
                                                    opts.w2v_path+opts.mem_w2v,ini_char,unk_char)

    print " post dict size:",post_ctable.getDictSize()
    print " post emb size:",post_ctable.getEmbSize()
    print " reply dict size:",reply_ctable.getDictSize()
    print " reply emb size:",reply_ctable.getEmbSize()
    print " mem dict size:",mem_ctable.getDictSize()
    print " mem emb size:",mem_ctable.getEmbSize()
    print ""

    memn2n_seq2seq = MemN2N_Seq2Seq(post_ctable.getDictSize(),mem_ctable.getDictSize(),reply_ctable.getDictSize(),opts.mem_size,
                                    post_ctable.getEmbSize(),opts.hidden_size,opts.batch_size,opts.hops,opts.max_length,
                                    opts.dropout,opts.teach_forcing,opts.memn2n_type,post_emb_matrix=post_ctable.getEmbMatrix(),
                                    mem_emb_matrix=mem_ctable.getEmbMatrix(),dec_emb_matrix=reply_ctable.getEmbMatrix()).cuda()

    if opts.weights != None:
        print "load model parameters..."
        memn2n_seq2seq.load_state_dict(torch.load(opts.weights))
    else:
        print "No model parameters!"
        exit()

    post_file_path = "/users3/yfwang/rg_data/corpus/ecm_filt_tab1000test.q"
    mem_file_path = "/users3/yfwang/rg_data/corpus/ecm_filt_tab1000test.all"

    print "start predicting..."
    reply_word2index = reply_ctable.getWord2Index()
    ini_idx = reply_word2index[ini_char]
    post_sentences,list_mem_sentences,decoded_sentences = predict_Sentences(post_ctable.getWord2Index(),mem_ctable.getWord2Index(),reply_ctable.getIndex2Word(),
                                                                            unk_char,ini_char,ini_idx,memn2n_seq2seq,post_file_path,mem_file_path,opts.print_every,
                                                                            opts.batch_size,opts.max_length,opts.mem_size)

    if opts.memn2n_type == 0:
        pred_file = open("pred_result_Adj_mem",'w')
        pred_a_file = open("pred_answer_Adj_mem",'w')
    else:
        pred_file = open("pred_result_Lay_mem",'w')
        pred_a_file = open("pred_answer_Lay_mem",'w')

    for idx in range(len(decoded_sentences)):
        post = post_sentences[idx].replace("\t",' ')
        reply = decoded_sentences[idx]
        pred_file.write(post+'\n')
        for mem_sentence in list_mem_sentences[idx]:
            pred_file.write('->'+mem_sentence.replace('\t',' ')+'\n')
        pred_file.write(reply+'\n')
        pred_a_file.write(reply+'\n')
        pred_file.write('-------------------------------------'+'\n')
    pred_file.close()
    pred_a_file.close()
    
