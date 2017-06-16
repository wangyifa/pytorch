# -*- coding: utf-8 -*-
import random
import re
import time
import torch 
#from torch.autograd import Variable
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
            line_list = line.strip('\n').strip().split(' ')
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

# 去掉中文语句中的标点符号
'''def normalizeString(s):
    s = re.sub(r"([.!?()])", r" \1", s)
    return s'''

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
    post_sentence = corpus_pair[0]
    reply_sentence = corpus_pair[1]
    mem = corpus_pair[2]

    post_sentence = paddingSenten(post_sentence,max_length)
    reply_sentence = paddingSenten(reply_sentence,max_length)
    tmp_mem = []
    for mem_sentence in mem:
        mem_sentence = paddingSenten(mem_sentence,max_length)
        tmp_mem.append(mem_sentence)

    padding_pair.append(post_sentence)
    padding_pair.append(reply_sentence)
    padding_pair.append(tmp_mem)
    return tuple(padding_pair)

def paddingPairs(corpus_pairs,max_length):
    padding_pairs = []
    for corpus_pair in corpus_pairs:
        corpus_pair = paddingPair(corpus_pair,max_length)
        padding_pairs.append(corpus_pair)
    return padding_pairs
#####################################################

# 读取语料的内容，预处理后存储到corpus_pairs里
# 每一个pair为一个tuple,包括对话的post,reply,memory的list
def readingCorpus(post_file, reply_file, mem_file, mem_size):
    PostFile = open(post_file,'r')
    ReplyFile = open(reply_file,'r')
    MemFile = open(mem_file,'r')
    t_MemFile = open(mem_file,'r')
    len_MemFile = len(t_MemFile.readlines())
    if len_MemFile%mem_size != 0:
        print "mem_size error!!!!!"
        print len_MemFile
        t_MemFile.close()
    t_MemFile.close()

    list_post = []
    list_reply = []
    list_mem = []
    tmp_mem = []
    for post_line in PostFile:
        list_post.append(post_line.strip('\n').strip())
    for reply_line in ReplyFile:
        list_reply.append(reply_line.strip('\n').strip())
    for idx, mem_line in enumerate(MemFile):
        tmp_mem.append(mem_line.strip('\n').strip())
        if (idx+1)%mem_size == 0:
            list_mem.append(tmp_mem)
            tmp_mem = []

    corpus_pairs = []
    corpus_pair = []
    for post,reply,mem in zip(list_post,list_reply,list_mem):
        #corpus_pair.append(normalizeString(post))
        #corpus_pair.append(normalizeString(reply))
        corpus_pair.append(post)
        corpus_pair.append(reply)
        corpus_pair.append(mem)
        corpus_pairs.append(tuple(corpus_pair))
        corpus_pair = []
    
    PostFile.close()
    ReplyFile.close()
    MemFile.close()
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

def filteringPair(word2index,corpus_pair,unk_char,ini_char):
    filtering_pair = []
    post_sentence = corpus_pair[0]
    reply_sentence = corpus_pair[1]
    mem = corpus_pair[2]
    post_sentence = filteringSenten(word2index,post_sentence,unk_char,ini_char)
    reply_sentence = filteringSenten(word2index,reply_sentence,unk_char,ini_char)
    tmp_mem = []
    for mem_sentence in mem:
        mem_sentence = filteringSenten(word2index,mem_sentence,unk_char,ini_char)
        tmp_mem.append(mem_sentence)
    filtering_pair.append(post_sentence)
    filtering_pair.append(reply_sentence)
    filtering_pair.append(tmp_mem)
    return tuple(filtering_pair)

def filteringPairs(word2index,corpus_pairs,unk_char,ini_char):
    filtering_pairs = []
    for corpus_pair in corpus_pairs:
        corpus_pair = filteringPair(word2index,corpus_pair,unk_char,ini_char)
        filtering_pairs.append(corpus_pair)
    return filtering_pairs
#####################################################

# 读取语料，构建词表的主函数
def readingData(corpus_w2v, post_file, reply_file, mem_file, ini_char, unk_char, mem_size, max_length):
    print "loading corpus..."
    corpus_pairs = readingCorpus(post_file, reply_file, mem_file, mem_size)

    corpus_ctable = readingW2v(corpus_w2v,ini_char,unk_char)
    # 后处理：替换语料中不在w2v词表的词和padding操作
    corpus_pairs = filteringPairs(corpus_ctable.getWord2Index(),corpus_pairs,corpus_ctable.getUnkChar(),corpus_ctable.getIniChar())
    corpus_pairs = paddingPairs(corpus_pairs,max_length)

    print " num of pair:",len(corpus_pairs)
    return corpus_ctable,corpus_pairs

# 读取w2v文件，构建emb矩阵
def readingW2v(corpus_w2v, ini_char, unk_char):
    print "loading word2vec..."
    corpus_ctable = W2vCharacterTable(corpus_w2v,ini_char,unk_char)

    return corpus_ctable

##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
#从pair的list构造batch,可选是否shuffle
#返回batch的list，每个batch也是pair的list
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
# 语料转换成索引组成的Tensor的tensor
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

def tensorFromSentence(word2index, sentence):
    indexes = indexesFromSentence(word2index,sentence)
    #indexes.append(EOS_token)
    tensor = torch.LongTensor(indexes).view(-1,1)
    #return tensor.cuda()
    return tensor

def tensorsFromMem(word2index,mem):
    list_mem_tensor = []
    for m in mem:
        mem_tensor = tensorFromSentence(word2index,m)
        list_mem_tensor.append(mem_tensor)
    return list_mem_tensor

def tensorsFromPair(word2index,pair):
    post_tensor = tensorFromSentence(word2index,pair[0])
    reply_tensor = tensorFromSentence(word2index,pair[1])
    mem = pair[2]
    list_mem_tensor = []
    for m in mem:
        mem_tensor = tensorFromSentence(word2index,m)
        list_mem_tensor.append(mem_tensor)
    return post_tensor,reply_tensor,list_mem_tensor

#####################################################
#读入句对batch(也是pair的list)的list
#返回post,reply以及memory的batch的list
#其中每个batch为Tensor的tensor
#size为batch_size*max_length
#每次返回memory的list,大小为mem_size,其中每个元素size为batch_size*max_length
def getTensorsPairsBatch(corpus_word2index,pairs_batches,mem_size):
    for batch in pairs_batches:
        list_p = []
        list_r = []
        list_m = [[] for i in range(mem_size)]
        mems_tensor_batch = []
        #print ""
        #print "len(batch):",len(batch)
        for pair in batch:
            tensor_q,tensor_r,list_tensor_m = tensorsFromPair(corpus_word2index,pair)
            list_p.append(tensor_q.t())
            list_r.append(tensor_r.t())
            for idx,tensor_m in enumerate(list_tensor_m):
                list_m[idx].append(tensor_m.t())

        for m in list_m:
            mem_tensor_batch = torch.cat(tuple(m),0)
            mems_tensor_batch.append(mem_tensor_batch)

        post_tensor_batch = torch.cat(tuple(list_p),0)
        reply_tensor_batch = torch.cat(tuple(list_r),0)
        
        yield post_tensor_batch, reply_tensor_batch, mems_tensor_batch

        del post_tensor_batch
        del reply_tensor_batch
        del mems_tensor_batch
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

#从memory的list构造batch
def buildingMemBatch(list_mem,batch_size):
    num_mems = len(list_mem)
    num_batches = upDiv(num_mems,batch_size)
    mem_batches = [[]]*num_batches
    tmp_batch = ['']*batch_size

    for i in range(num_mems):
        idx_b = i/batch_size
        idx_in_b = i%batch_size
        tmp_batch[idx_in_b] = list_mem[i]
        if idx_in_b+1 == batch_size or i+1 == num_mems:
            if idx_in_b+1 < batch_size:
                for j in range(idx_in_b+1,batch_size):
                    tmp_batch[j] = tmp_batch[idx_in_b]
            mem_batches[idx_b] = tmp_batch
            tmp_batch = ['']*batch_size

    return mem_batches

#####################################################
#读入batch的list
#将每个句子list的batch转换成为Tensor的tensor
#size为max_length*batch_size 
def getTensorsSentenMemBatch(corpus_word2index,sentence_batches,mem_batches,mem_size):
    for sentence_batch,mem_batch in zip(sentence_batches,mem_batches):
        list_s = []
        list_m = [[] for i in range(mem_size)]
        mems_tensor_batch = []
        for sentence in sentence_batch:
            tensor_sentence = tensorFromSentence(corpus_word2index,sentence)
            list_s.append(tensor_sentence.t())

        for mem in mem_batch:
            list_tensor_m = tensorsFromMem(corpus_word2index,mem)
            for idx,tensor_m in enumerate(list_tensor_m):
                list_m[idx].append(tensor_m.t())
        
        for m in list_m:
            mem_tensor_batch = torch.cat(tuple(m),0)
            mems_tensor_batch.append(mem_tensor_batch)
        sentence_tensor_batch = torch.cat(tuple(list_s),0)

        yield sentence_tensor_batch,mems_tensor_batch
        del sentence_tensor_batch
        del mems_tensor_batch
        gc.collect()

if __name__ == '__main__':
    corpus_w2v = "/users3/yfwang/rg_data/w2v/jx1log_tab_all.w2v"

    post_file = "/users3/yfwang/rg_data/corpus/jx1log_tab10.q"
    reply_file = "/users3/yfwang/rg_data/corpus/jx1log_tab10.a"
    mem_file = "/users3/yfwang/rg_data/corpus/jx1log_all_tab10.mem"
    
    corpus_ctable,corpus_pairs = readingData(corpus_w2v, post_file, reply_file, mem_file, ini_char='</i>', 
                                            unk_char='<unk>', mem_size=4, max_length=10)
    print " corpus dict size:",corpus_ctable.getDictSize()
    print " corpus emb size:",corpus_ctable.getEmbSize()

    corpus_word2index = corpus_ctable.getWord2Index()
    corpus_index2word = corpus_ctable.getIndex2Word()
    corpus_emb_matrix = corpus_ctable.getEmbMatrix()

    unk_index = corpus_word2index['<unk>']
    unk_emb = corpus_emb_matrix[unk_index]

    pairs_batches,num_batches = buildingPairsBatch(corpus_pairs,batch_size=1,shuffle=True)
    print " num_batches:",num_batches

    pairs_batch = pairs_batches[0]
    print type(pairs_batch)

   
    '''for post_tensor_batch,reply_tensor_batch,mems_tensor_batch in getTensorsPairsBatch(post_word2index,reply_word2index,
                                                                mem_word2index,pairs_batches,mem_size=4):
        print ""
        #print "post_tensor_batch.size:",post_tensor_batch.size()
        #print "reply_tensor_batch.size:",reply_tensor_batch.size()
        #print "mem_tensor_batch.size:",mems_tensor_batch[0].size()
        print type(post_tensor_batch)
        print type(reply_tensor_batch)
        print type(mems_tensor_batch[0])
        #del post_tensor_batch
        #del reply_tensor_batch
        #del mems_tensor_batch
        #gc.collect()
        #time.sleep(0.1)

    x = raw_input()'''