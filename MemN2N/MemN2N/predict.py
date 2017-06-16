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
    description.add_argument("--corpus_w2v", type=str, default="jx1log_tab_all.w2v")
    
    description.add_argument("--memn2n_type", type=int, default=0)
    description.add_argument("--mem_size", type=int, default=4)
    description.add_argument("--batch_size", type=int, default=32)
    description.add_argument("--hidden_size", type=int, default=1024)
    description.add_argument("--max_length", type=int, default=10)
    description.add_argument("--hops", type=int, default=3)

    description.add_argument("--dropout", type=float, default=0.25)
    description.add_argument("--teach_forcing", type=int, default=1)
    description.add_argument("--print_every", type=int, default=100, help="print every batches when training")

    description.add_argument("--weights", type=str, default=None)
    return description.parse_args(argv)

opts = init_command_line(sys.argv[1:])
print "Configure:"
print " corpus_w2v:",opts.w2v_path + opts.corpus_w2v

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
def predict_Sentences(corpus_word2index,corpus_index2word,unk_char,ini_char,ini_idx,model,
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
        sentence = filteringSenten(corpus_word2index,sentence,unk_char,ini_char)
        sentence = paddingSenten(sentence,max_length)
        list_pad_post.append(sentence)

    tmp_mem = []
    list_pad_mem = []
    for memidx,sentence in enumerate(mem_sentences):
        sentence = filteringSenten(corpus_word2index,sentence,unk_char,ini_char)
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
    for post_tensor_batch,mems_tensor_batch in getTensorsSentenMemBatch(corpus_word2index,post_batches,mem_batches,mem_size):
        predict_batch = model.predict(post_tensor_batch,mems_tensor_batch,corpus_index2word,ini_idx,sep_char=' ')
        predict_sentences.extend(predict_batch)
        if (idx_batch+1)%print_every == 0:
            print "{} batches finished".format(idx_batch+1)
        idx_batch += 1

    predict_sentences = predict_sentences[0:num_post]
    return post_sentences,list_pad_mem,predict_sentences

if __name__ == '__main__':
    ini_char = '</i>'
    unk_char = '<unk>'
    corpus_ctable = readingW2v(opts.w2v_path+opts.corpus_w2v, ini_char, unk_char)

    print " corpus dict size:",corpus_ctable.getDictSize()
    print " corpus emb size:",corpus_ctable.getEmbSize()
    print ""

    memn2n_seq2seq = MemN2N_Seq2Seq(corpus_ctable.getDictSize(),opts.mem_size,corpus_ctable.getEmbSize(),opts.hidden_size,
                                    opts.batch_size,opts.hops,opts.max_length,opts.dropout,opts.teach_forcing,opts.memn2n_type,
                                    emb_matrix=corpus_ctable.getEmbMatrix()).cuda()

    if opts.weights != None:
        print "load model parameters..."
        memn2n_seq2seq.load_state_dict(torch.load(opts.weights))
    else:
        print "No model parameters!"
        exit()

    post_file_path = "/users3/yfwang/rg_data/corpus/jx1log_tab_200test.q"
    mem_file_path = "/users3/yfwang/rg_data/corpus/jx1log_all_tab_200test.mem"

    print "start predicting..."
    corpus_word2index = corpus_ctable.getWord2Index()
    ini_idx = corpus_word2index[ini_char]
    post_sentences,list_mem_sentences,decoded_sentences = predict_Sentences(corpus_word2index,corpus_ctable.getIndex2Word(),unk_char,ini_char,
                                                                            ini_idx,memn2n_seq2seq,post_file_path,mem_file_path,opts.print_every,
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
    
