# -*- coding: utf-8 -*-
import sys
import random
import re
import time
import torch 
from torch.autograd import Variable
import torch.nn.init as init
import gc

# word2vec类
class W2vCharacterTable(object):
    def __init__(self, w2v_path, ini_char = '</i>', unk_char='<unk>'):
        super(W2vCharacterTable, self).__init__()
        w2vf = open(w2v_path,'r')
        self._dict_size = 0
        self._emb_size = 0
        self._word2index = {}
        self._index2word = {}
        self._ini_char = ini_char 
        self._unk_char = unk_char

        list_emb = []
        idx = 0
        for line in w2vf:
            line_list = line.strip().split(' ')
            if len(line_list) == 2:
                # 从w2v文件第一行获取word2vec的dict大小和embedding维度
                self._dict_size = int(line_list[0])
                self._emb_size = int(line_list[1])
            else:
                # 读取word和emb
                self._word2index[line_list[0]] = idx
                self._index2word[idx] = line_list[0]
                tmp_emb = [float(item) for item in line_list[1:]]
                list_emb.append(torch.FloatTensor(tmp_emb).unsqueeze(0).cuda())
                idx += 1
        w2vf.close()

        if not self._word2index.has_key(self._unk_char):
            self._word2index[self._unk_char] = self._dict_size
            self._index2word[self._dict_size] = self._unk_char
            self._dict_size += 1
            list_emb.append(init.uniform(torch.FloatTensor(1,self._emb_size).cuda(),-0.1,0.1))

        if not self._word2index.has_key(self._ini_char):
            self._word2index[self._ini_char] = self._dict_size
            self._index2word[self._dict_size] = self._ini_char
            self._dict_size += 1
            list_emb.append(init.uniform(torch.FloatTensor(1,self._emb_size).cuda(),-0.1,0.1))

        # 构建embedding的大矩阵
        self._emb_matrix = torch.cat(tuple(list_emb),0)
        
    def getDictSize(self):
        return self._dict_size

    def getEmbSize(self):
        return self._emb_size

    def getEmbMatrix(self):
        return self._emb_matrix

    def getWord2Index(self):
        return self._word2index

    def getIndex2Word(self):
        return self._index2word

    def getIniChar(self):
        return self._ini_char

    def getUnkChar(self):
        return self._unk_char

#####################################################
#padding 操作
def paddingSenten(sentence,max_length):
    ListSenten = sentence.split('\t')
    if len(ListSenten) <= max_length:
        ListSenten.extend(["</s>"]*(max_length-len(ListSenten)))
    else:
        ListSenten = ListSenten[0:max_length]
    return '\t'.join(ListSenten)

def paddingPair(corpus_pair,max_length):
    padding_pair = []
    for sentence in corpus_pair:
        sentence = paddingSenten(sentence,max_length)
        padding_pair.append(sentence)
    return padding_pair

def paddingPairs(corpus_pairs,max_length):
    padding_pairs = []
    for corpus_pair in corpus_pairs:
        corpus_pair = paddingPair(corpus_pair,max_length)
        padding_pairs.append(corpus_pair)
    return padding_pairs
#####################################################

# 读取语料的内容，预处理后存储到corpus_pairs里
def readingCorpus(post_file, reply_file):
    PostFile = open(post_file,'r')
    ReplyFile = open(reply_file,'r')
    list_post = []
    list_reply = []
    for post_line in PostFile:
        list_post.append(post_line.strip())
    for reply_line in ReplyFile:
        list_reply.append(reply_line.strip())

    corpus_pairs = []
    corpus_pair = []
    for post,reply in zip(list_post,list_reply):
        corpus_pair.append(post)
        corpus_pair.append(reply)
        corpus_pairs.append(corpus_pair)
        corpus_pair = []
 
    return corpus_pairs

#####################################################
# 未登录词替换成unk_char
def filteringSenten(word2index,sentence,unk_char,ini_char):
    ListSenten = sentence.split('\t')
    filteringListSenten = []
    for word in ListSenten:
        if word2index.has_key(word):
            filteringListSenten.append(word)
        else:
            filteringListSenten.append(unk_char)
    # 若过滤后句子为空:
    if filteringListSenten == []:
        filteringListSenten = [ini_char]
    return '\t'.join(filteringListSenten)

def filteringPair(post_word2index,reply_word2index,corpus_pair,unk_char,ini_char):
    filtering_pair = []
    post_sentence = corpus_pair[0]
    reply_sentence = corpus_pair[1]
    post_sentence = filteringSenten(post_word2index,post_sentence,unk_char,ini_char)
    reply_sentence = filteringSenten(reply_word2index,reply_sentence,unk_char,ini_char)
    filtering_pair.append(post_sentence)
    filtering_pair.append(reply_sentence)
    return filtering_pair

def filteringPairs(post_word2index,reply_word2index,corpus_pairs,unk_char,ini_char):
    filtering_pairs = []
    for corpus_pair in corpus_pairs:
        corpus_pair = filteringPair(post_word2index,reply_word2index,corpus_pair,unk_char,ini_char)
        filtering_pairs.append(corpus_pair)
    return filtering_pairs
#####################################################

# 读取语料，构建词表的主函数
def readingData(post_w2v, reply_w2v, post_file, reply_file, ini_char, unk_char, max_length):
    print "loading corpus..."
    corpus_pairs = readingCorpus(post_file, reply_file)

    post_ctable,reply_ctable = readingW2v(post_w2v,reply_w2v,ini_char,unk_char,)
    # 后处理：去掉语料中不在w2v词表的词和padding操作
    corpus_pairs = filteringPairs(post_ctable.getWord2Index(),reply_ctable.getWord2Index(),
                                    corpus_pairs,post_ctable.getUnkChar(),post_ctable.getIniChar())
    corpus_pairs = paddingPairs(corpus_pairs,max_length)

    print " num of pair:",len(corpus_pairs)
    return post_ctable,reply_ctable,corpus_pairs

# 读取w2v文件，构建emb矩阵
def readingW2v(post_w2v, reply_w2v, ini_char, unk_char):
    print "loading word2vec..."
    post_ctable = W2vCharacterTable(post_w2v,ini_char,unk_char)
    reply_ctable = W2vCharacterTable(reply_w2v,ini_char,unk_char)

    return post_ctable,reply_ctable

#####################################################
#从pair的list构造batch,可选是否shuffle
#返回batch的list，每个batch也是句pair的list
#若最后一个batch大小与之前不同，会用最后一个元素进行padding
def buildingPairsBatch(corpus_pairs,batch_size,shuffle=True):
    if shuffle:
        print "shuffle..."
        #t0 = time.time()
        random.shuffle(corpus_pairs)
        #print "shuffle time:",time.time()-t0
    pairs_batches = []
    batch = []
    for idx,pair in enumerate(corpus_pairs):
        batch.append(pair)
        if (idx+1)%batch_size == 0 or (idx+1)==len(corpus_pairs):
            pairs_batches.append(batch)
            batch = []
    num_batches = len(pairs_batches)

    # padding
    last_batch = pairs_batches[-1]
    if len(last_batch) < batch_size:
        for i in range(len(last_batch),batch_size):
            pairs_batches[-1].append(last_batch[len(last_batch)-1])

    return pairs_batches,num_batches

#####################################################
# 语料转换成索引组成的Tensor的variable
def indexesFromSentence(word2index, sentence):
    indexes = []
    for word in sentence.split('\t'):
        try:
            index = word2index[word]
            indexes.append(index)
        except Exception:
            print "sentence:",sentence
            if word == '':
                print "Error!"
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print exc_type,"[",exc_traceback.tb_lineno,"]",str(exc_value)
    return indexes

def tensorFromSentence(word2index,sentence):
    indexes = indexesFromSentence(word2index,sentence)
    #indexes.append(EOS_token)
    tensor = torch.LongTensor(indexes).view(-1,1)
    return tensor

def tensorsFromPair(post_word2index,reply_word2index,pair):
    post_tensor = tensorFromSentence(post_word2index,pair[0])
    reply_tensor = tensorFromSentence(reply_word2index,pair[1])
    return post_tensor,reply_tensor

#####################################################
def sentenFromTensor(index2word,tensor):
    sentence = []
    for idx in tensor:
        word = index2word[idx]
        sentence.append(word)
    return sentence

#####################################################
#读入句对batch(也是pair的list)的list
#返回源句和目标句的batch的list
#其中每个batch为Tensor的variable
#size为max_length*batch_size 
def getTensorsPairsBatch(post_word2index,reply_word2index,pairs_batches):
    for batch in pairs_batches:
        list_p = []
        list_r = []
        #print ""
        #print "len(batch):",len(batch)
        for pair in batch:
            post_tensor,reply_tensor = tensorsFromPair(post_word2index,reply_word2index,pair)
            list_p.append(post_tensor)
            list_r.append(reply_tensor)

        post_tensor_batch = torch.cat(tuple(list_p),1)
        reply_tensor_batch = torch.cat(tuple(list_r),1)
        
        yield post_tensor_batch, reply_tensor_batch
        del post_tensor_batch
        del reply_tensor_batch
        gc.collect()

#####################################################
# 向上取整的除法
def upDiv(divisor,dividend):
    return int((divisor+dividend-1)/dividend) 

#从句子的list构造batch
#返回batch的list，每个batch也是句子的list
#若最后一个batch大小与之前不同，会用最后一个元素进行padding
def buildingSentenBatch(list_senten,batch_size):
    num_sentences = len(list_senten)
    num_batches = upDiv(num_sentences,batch_size)
    sentence_batches = [[]]*num_batches
    tmp_batch = ['']*batch_size

    for i in range(num_sentences):
        idx_b = i/batch_size
        idx_in_b = i%batch_size
        tmp_batch[idx_in_b] = list_senten[i]
        if idx_in_b+1 == batch_size or i+1 == num_sentences:
            if idx_in_b+1 < batch_size:
                for j in range(idx_in_b+1,batch_size):
                    tmp_batch[j] = tmp_batch[idx_in_b]
            sentence_batches[idx_b] = tmp_batch
            tmp_batch = ['']*batch_size
    '''for batch in sentence_batches:
        for s in batch:
            print s
        print "----"'''
    return sentence_batches

#####################################################
#读入batch的list
#将每个句子list的batch转换成为Tensor的variable
#size为max_length*batch_size 
def getTensorsSentenBatch(word2index,sentence_batches):
    for batch in sentence_batches:
        list_s = []
        for sentence in batch:
            tensor_sentence = tensorFromSentence(word2index,sentence)
            list_s.append(tensor_sentence)

        tensor_batch = torch.cat(tuple(list_s),1)
       
        yield tensor_batch
        del tensor_batch
        gc.collect()

def getSentenTensorsBatch(index2word,tensors_batch):
    list_senten = []
    for tensor in tensors_batch.t():
        sentence = sentenFromTensor(index2word,tensor)
        list_senten.append(sentence)

    return list_senten

if __name__ == '__main__':
    post_w2v = "/users3/yfwang/rg_data/w2v/ecm_tab_q.w2v"
    reply_w2v = "/users3/yfwang/rg_data/w2v/ecm_tab_a.w2v"
    post_file = "/users3/yfwang/rg_data/corpus/ecm_tab.q"
    reply_file = "/users3/yfwang/rg_data/corpus/ecm_tab.a"
    batch_size = 64

    post_ctable,reply_ctable,corpus_pairs = readingData(post_w2v, reply_w2v, post_file, reply_file, ini_char='</i>',
                                                        unk_char='<unk>', max_length=10)

    print " post dict size:",post_ctable.getDictSize()
    print " post emb size:",post_ctable.getEmbSize()
    print " reply dict size:",reply_ctable.getDictSize()
    print " reply emb size:",reply_ctable.getEmbSize()
    print ""

    post_word2index = post_ctable.getWord2Index()
    post_index2word = post_ctable.getIndex2Word()
    post_emb_matrix = post_ctable.getEmbMatrix()

    reply_word2index = reply_ctable.getWord2Index()
    reply_index2word = reply_ctable.getIndex2Word()
    reply_emb_matrix = reply_ctable.getEmbMatrix()

    ini_idx = reply_word2index['</i>']
    print "idx of ini_char:",ini_idx
    stop_idx = reply_word2index['</s>']
    print "idx of stop_char:",stop_idx
    unk_idx = reply_word2index['<unk>']
    print "idx of unk_char:",unk_idx
    print ""

    pairs_batches,num_batches = buildingPairsBatch(corpus_pairs,batch_size,shuffle=True)
    print "num_batches:",num_batches

    for post_tensor_batch,reply_tensor_batch in getTensorsPairsBatch(post_word2index,reply_word2index,pairs_batches):
        print ""
        print "post_tensor_batch.size:",post_tensor_batch.size()
        print "reply_tensor_batch.size:",reply_tensor_batch.size()
        break

    #x = raw_input()