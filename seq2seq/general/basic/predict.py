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
from model import *
from data_utils import *

def init_command_line(argv):
    from argparse import ArgumentParser
    usage = "seq2seq"
    description = ArgumentParser(usage)
    description.add_argument("--w2v_path", type=str, default="/users3/yfwang/rg_data/w2v/")
    description.add_argument("--post_w2v", type=str, default="ecm_filt_tab_q_500e.w2v")
    description.add_argument("--reply_w2v", type=str, default="ecm_filt_tab_a_500e.w2v")
    
    description.add_argument("--batch_size", type=int, default=64)
    description.add_argument("--hidden_size", type=int, default=1024)
    description.add_argument("--num_layers", type=int, default=1)
    description.add_argument("--max_length", type=int, default=10)
    description.add_argument("--dropout", type=float, default=0.25)
    description.add_argument("--teach_forcing", type=int, default=1) 
    
    description.add_argument("--print_every", type=int, default=500, help="print every batches when predicting")
    description.add_argument("--weights", type=str, default=None)

    return description.parse_args(argv)
    
opts = init_command_line(sys.argv[1:])
print "Configure:"
print " post_w2v:",opts.w2v_path + opts.post_w2v
print " reply_w2v:",opts.w2v_path + opts.reply_w2v

print " batch_size:",opts.batch_size
print " hidden_size:",opts.hidden_size
print " num_layers:",opts.num_layers
print " max_length:",opts.max_length
print " dropout:",opts.dropout
print " teach_forcing:",opts.teach_forcing

print " print_every:",opts.print_every
print " weights:",opts.weights
print ""

# 读入一个句子的list，构建batch后进行预测
def predict_Sentences(post_word2index,reply_index2word,unk_char,ini_char,ini_idx,
                        model,post_file_path,print_every,batch_size,max_length):
    model.eval()

    post_file = open(post_file_path,'r') 
    post_sentences = []
    for line in post_file:
        line = line.strip()
        list_t = line.split('\t')
        post_sentences.append(line)
    post_file.close()
    
    list_pad_senten = []
    for sentence in post_sentences:
        sentence = filteringSenten(post_word2index,sentence,unk_char,ini_char)
        sentence = paddingSenten(sentence,max_length)
        list_pad_senten.append(sentence)
    
    #构造batch的list
    num_sentences = len(list_pad_senten)
    sentence_batches = buildingSentenBatch(list_pad_senten,batch_size)
    print "num of batch:",len(sentence_batches)
    print ""
    predict_sentences = []
    idx_batch = 0 
    for post_batch in getTensorsSentenBatch(post_word2index,sentence_batches):
        predict_batch = model.predict(post_batch,reply_index2word,ini_idx,sep_char=' ')
        predict_sentences.extend(predict_batch)
        if (idx_batch+1)%print_every == 0:
            print "{} batches finished".format(idx_batch+1)
        idx_batch += 1

    predict_sentences = predict_sentences[0:num_sentences]
    return post_sentences,predict_sentences

if __name__ == '__main__':
    ini_char = '</i>'
    unk_char = '<unk>'
    post_ctable,reply_ctable = readingW2v(opts.w2v_path+opts.post_w2v, opts.w2v_path+opts.reply_w2v, ini_char, unk_char)

    print " post dict size:",post_ctable.getDictSize()
    print " post emb size:",post_ctable.getEmbSize()
    print " reply dict size:",reply_ctable.getDictSize()
    print " reply emb size:",reply_ctable.getEmbSize()
    print ""

    seq2seq_model = Seq2Seq(post_ctable.getEmbMatrix(),reply_ctable.getEmbMatrix(),post_ctable.getDictSize(),
                            reply_ctable.getDictSize(),post_ctable.getEmbSize(),opts.hidden_size,opts.batch_size,
                            opts.num_layers,opts.dropout,opts.max_length,opts.teach_forcing).cuda()
    if opts.weights != None:
        print "load model parameters..."
        seq2seq_model.load_state_dict(torch.load(opts.weights))
    else:
        print "No model parameters!"
        exit()

    post_file_path = "/users3/yfwang/rg_data/corpus/test_tab_post.txt"
    print "start predicting..."

    reply_word2index = reply_ctable.getWord2Index()
    ini_idx = reply_word2index[ini_char]
    post_sentences,decoded_sentences = predict_Sentences(post_ctable.getWord2Index(),reply_ctable.getIndex2Word(),post_ctable.getUnkChar(),
                                                        post_ctable.getIniChar(),ini_idx,seq2seq_model,post_file_path,opts.print_every,
                                                        opts.batch_size,opts.max_length)

    pred_file = open("pred_result_basic",'w')
    pred_a_file = open("pred_answer_basic",'w')
    for idx in range(len(decoded_sentences)):
        post = post_sentences[idx].replace("\t",' ')
        reply = decoded_sentences[idx]
        pred_file.write(post+'\n')
        pred_file.write(reply+'\n')
        pred_a_file.write(reply+'\n')
        pred_file.write('-------------------------------------'+'\n')
    pred_file.close()
    pred_a_file.close()
    
