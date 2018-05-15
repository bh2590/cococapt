#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 17:14:34 2018

@author: hanozbhathena
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from utils import SpecialTokens, MyCOCODset, make_caption_word_dict, loadWordVectors
import numpy as np
import torchvision.models as models
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm

#torch.manual_seed(1)

from ipdb import slaunch_ipdb_on_exception
import ipdb as pdb

class Decoder(nn.Module):
    def __init__(self, emb_mat, vocab_size, word_vec_size, num_layers, output_size, word_to_idx,
                 dropout= 0.5, bidirectional= False, trainable_embeddings= False, batch_first= True,
                 max_seq_len=20):
        super(Decoder, self).__init__()
        self.batch_first= batch_first
        self.max_seq_len= max_seq_len
        self.emb_layer= nn.Embedding(vocab_size, word_vec_size)
        self.emb_layer.weight= nn.Parameter(torch.from_numpy(emb_mat.astype(np.float32)), 
                                            requires_grad= trainable_embeddings)
        self.gru_layer= nn.GRU(input_size= word_vec_size, hidden_size= output_size, num_layers= num_layers,
                           dropout= dropout, batch_first= self.batch_first, bidirectional= bidirectional)
        self.vocab_project= nn.Linear(output_size, vocab_size)
        self.start_id= torch.tensor(word_to_idx[SpecialTokens.START], dtype= torch.long)
    
    def forward(self, img_rep, targets, real_lens, is_train= True):
        pdb.set_trace()
        start_batch= self.start_id.repeat(targets.size()[0], 1)
        if is_train == True: #replace with self.training later
            teacher_inps= torch.cat([start_batch, targets[:, :-1]], dim= 1)
            emb_inps= self.emb_layer(teacher_inps)
#            real_lens+= 1
            real_lens_sorted, idx = real_lens.sort(0, descending=True)
            emb_inps_sorted = emb_inps[idx]
            packed_seq_x= pack_padded_sequence(emb_inps_sorted, real_lens_sorted, batch_first= self.batch_first)
            h0= img_rep.unsqueeze(0)
            packed_out, packed_h_t= self.gru_layer(packed_seq_x, h0)
            unpacked_out, _= pad_packed_sequence(packed_out, batch_first= self.batch_first, 
                                                 total_length= targets.size()[1])
            _, orig_idx = idx.sort(0)
            final_out = unpacked_out[orig_idx]
            return self.vocab_project(final_out)
    
    def inference(self, img_rep):
        inp= self.start_id.repeat(img_rep.size()[0], 1)
        state= img_rep.unsqueeze(0)
        word_ind_list= []
        for step in range(self.max_seq_len):
            output, state= self.gru_layer(inp, state)
            logits= self.vocab_project(output)
            m= torch.distributions.categorical.Categorical(logits= logits)
            word_inds= m.sample()
            word_ind_list.append(word_inds)
            inp= output
        

#    def forward(self, features, captions, lengths):
#        """Decode image feature vectors and generates captions."""
#        pdb.set_trace()
#        embeddings = self.emb_layer(captions)
#        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
#        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
#        hiddens, _ = self.gru(packed)
#        outputs = self.vocab_project(hiddens[0])
#        return outputs


def fn(model, x):
    x= model.conv1(x)
    x= model.bn1(x)
    x= model.relu(x)
    x= model.maxpool(x)
    return x

class Encoder(nn.Module):
    def __init__(self, model, img_feature_size, dec_hidden_size):
        super(Encoder, self).__init__()
        self.model= model
        self.linear= nn.Linear(img_feature_size, dec_hidden_size)
#        self.linear= nn.Linear(img_feature_size, 300)
    
    def forward(self, x):
        pdb.set_trace()
        x= fn(self.model, x)
        x= x.view(x.size()[0], -1)
        x= self.linear(x)
        return x


class ImageCaption(nn.Module):
    def __init__(self, encoder, decoder):
        self.encoder= encoder
        self.decoder= decoder
    
    def forward(self, img, caption, real_lens):
        x= self.encoder(img)
        predictions= self.decoder(x, caption, real_lens)
        return predictions


