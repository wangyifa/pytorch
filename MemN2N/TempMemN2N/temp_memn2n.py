# -*- coding: utf-8 -*-
import time
import math
import torch 
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class TempAjcnMemN2N(nn.Module):
    """End-to-End Memory Network (Adjacent)  https://arxiv.org/abs/1503.08895
    u = B*q, m_i = A*x_i + TA(i), c_i = C*x_i + TC(i)
    p_i = softmax(u*m_i), o = ∑(p_i*c_i)
    u_k+1 = u_k + o_k, a = softmax(W*u)
    其中 A_k+1 = C_k, TA_k+1 = TC_k, 
    d: hidden_size, V: emb_size"""
    def __init__(self, mem_size, hidden_size, emb_size, batch_size, hops):
        super(TempAjcnMemN2N, self).__init__()
        self.mem_size = mem_size
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.hops = hops

        self.A_1 = nn.Parameter(torch.FloatTensor(self.emb_size, self.hidden_size))
        self.C_k = nn.Parameter(torch.FloatTensor(self.emb_size, self.hidden_size))
        self.B = nn.Parameter(torch.FloatTensor(self.emb_size, self.hidden_size))

        self.TA_1 = nn.Parameter(torch.FloatTensor(self.mem_size, self.hidden_size))
        self.TC_k = nn.Parameter(torch.FloatTensor(self.mem_size, self.hidden_size))

        '''参数A_2,A_3,...A_k 其中A_k+1 = C_k 
        weights[i] = A_(i+2) = C_(i+1)
        TA_2,TA_3,...TA_k 其中TA_k+1 = TC_k 
        T_weights[i] = TA_(i+2) = TC_(i+1)'''
        self.weights = nn.ParameterList([nn.Parameter(torch.FloatTensor(self.emb_size, self.hidden_size)) for i in range(self.hops-1)])
        self.T_weights = nn.ParameterList([nn.Parameter(torch.FloatTensor(self.mem_size, self.hidden_size)) for i in range(self.hops-1)])

    def forward(self,inputs,questions):
        '''question: batch_size*max_length*emb_size
        B: batch_size*emb_size*hidden_size
        u: batch_size*1*hidden_size'''
        B = torch.stack((self.B,)*self.batch_size,0)
        u = torch.sum(torch.bmm(questions,B),dim=1)

        for i in range(1,self.hops+1):
            mem_m = [None]*self.mem_size
            mem_c = [None]*self.mem_size
            mem_p = [None]*self.mem_size
            '''inputs[j]: batch_size*max_length*emb_size
            A: batch_size*emb_size*hidden_size
            TA_j: batch_size*1*hidden_size
            m: batch_size*1*hidden_size'''
            for j in range(self.mem_size):
                if i == 1:
                    A = torch.stack((self.A_1,)*self.batch_size,0)
                    TA_j = torch.stack((self.TA_1[j],)*self.batch_size,0)
                else:
                    A = torch.stack((self.weights[i-2],)*self.batch_size,0)
                    TA_j = torch.stack((self.T_weights[i-2][j],)*self.batch_size,0)
                m = torch.sum(torch.bmm(inputs[j],A),dim=1) + TA_j
                mem_m[j] = m

            '''C: batch_size*emb_size*hidden_size
            TC_j: batch_size*1*hidden_size
            c: batch_size*1*hidden_size'''
            for j in range(self.mem_size):
                if i == self.hops:
                    C = torch.stack((self.C_k,)*self.batch_size,0)
                    TC_j = torch.stack((self.TC_k[j],)*self.batch_size,0)
                else:
                    C = torch.stack((self.weights[i-1],)*self.batch_size,0)
                    TC_j = torch.stack((self.T_weights[i-1][j],)*self.batch_size,0)
                c = torch.sum(torch.bmm(inputs[j],C),dim=1) + TC_j
                mem_c[j] = c
            
            '''u: batch_size*1*hidden_size
            p_j: batch_size*1; p:batch_size*1*mem_size
            o: batch_size*1*hidden_size'''
            for j in range(self.mem_size):
                p_j = torch.bmm(u,mem_m[j].transpose(1,2)).squeeze(2)
                mem_p[j] = p_j
            p = F.softmax(torch.cat(tuple(mem_p),1)).unsqueeze(1)

            o = torch.bmm(p,torch.cat(tuple(mem_c),1))
            u = u+o

        return u

class TempLayerMemN2N(nn.Module):
    """End-to-End Memory Network (Layer-wise/RNN-like)  https://arxiv.org/abs/1503.08895
    u = B*q, m_i = A*x_i + TA(i), c_i = C*x_i + TC(i)
    p_i = softmax(u*m_i), o = ∑(p_i*c_i)
    u_k+1 = H*u_k + o_k, a = softmax(W*u_k+1))
    其中 A_1 = A_2 = ... = A_k, TA_1 = TA_2 = ... = TA_k 
    C_1 = C_2 = ... = C_k, TC_1 = TC_2 = ... = TC_k 
    d: hidden_size, V: emb_size"""
    def __init__(self, mem_size, hidden_size, emb_size, batch_size, hops):
        super(TempLayerMemN2N, self).__init__()
        self.mem_size = mem_size
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.hops = hops

        '''A_1 = A_2 = ... = A_k,  
        C_1 = C_2 = ... = C_k, 
        TA_1 = TA_2 = ... = TA_k
        TC_1 = TC_2 = ... = TC_k '''
        self.A = nn.Parameter(torch.FloatTensor(self.emb_size, self.hidden_size))
        self.C = nn.Parameter(torch.FloatTensor(self.emb_size, self.hidden_size))
        self.B = nn.Parameter(torch.FloatTensor(self.emb_size, self.hidden_size))

        self.TA = nn.Parameter(torch.FloatTensor(self.mem_size, self.hidden_size))
        self.TC = nn.Parameter(torch.FloatTensor(self.mem_size, self.hidden_size))

    def forward(self,inputs,questions):
        '''question: batch_size*max_length*emb_size
        B: batch_size*emb_size*hidden_size
        u: batch_size*1*hidden_size'''
        B = torch.stack((self.B,)*self.batch_size,0)
        u = torch.sum(torch.bmm(questions,B),dim=1)

        for i in range(1,self.hops+1):
            mem_m = [None]*self.mem_size
            mem_c = [None]*self.mem_size
            mem_p = [None]*self.mem_size
            '''input: batch_size*max_length*emb_size
            A: batch_size*emb_size*hidden_size
            TA_j: batch_size*1*hidden_size
            m: batch_size*1*hidden_size'''
            for j in range(self.mem_size):
                A = torch.stack((self.A,)*self.batch_size,0)
                TA_j = torch.stack((self.TA[j],)*self.batch_size,0)
                m = torch.sum(torch.bmm(inputs[j],A),dim=1) + TA_j
                mem_m[j] = m

            '''C: batch_size*emb_size*hidden_size
            TC_j: batch_size*1*hidden_size
            c: batch_size*1*hidden_size'''
            for j in range(self.mem_size):
                C = torch.stack((self.C,)*self.batch_size,0)
                TC_j = torch.stack((self.TC[j],)*self.batch_size,0)
                c = torch.sum(torch.bmm(inputs[j],C),dim=1) + TC_j
                mem_c[j] = c

            '''u: batch_size*1*hidden_size
            p_j: batch_size*1; p:batch_size*1*mem_size
            o: batch_size*1*hidden_size'''
            for j in range(self.mem_size):
                p_j = torch.bmm(u,mem_m[j].transpose(1,2)).squeeze(2)
                mem_p[j] = p_j
            p = F.softmax(torch.cat(tuple(mem_p),1)).unsqueeze(1)

            o = torch.bmm(p,torch.cat(tuple(mem_c),1))
            u = u+o

        return u