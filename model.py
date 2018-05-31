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
        self.num_layers= num_layers
        self.bidirectional= bidirectional
        self.num_directions= 1 if bidirectional == False else 2
        self.output_size= output_size
        self.emb_layer= nn.Embedding(vocab_size, word_vec_size)
        self.emb_layer.weight= nn.Parameter(torch.from_numpy(emb_mat.astype(np.float32)), 
                                            requires_grad= trainable_embeddings)
        self.gru_layer= nn.GRU(input_size= word_vec_size, hidden_size= output_size, num_layers= num_layers,
                           dropout= dropout, batch_first= self.batch_first, bidirectional= bidirectional)
        self.vocab_project= nn.Linear(output_size, vocab_size)
        self.start_id= torch.tensor(word_to_idx[SpecialTokens.START], dtype= torch.long)
    
    def forward(self, img_rep, targets, real_lens, is_train= True):
#        pdb.set_trace()
        start_batch= img_rep.unsqueeze(1)
        if is_train == True: #replace with self.training later
            teacher_inps= targets[:, :-1]
            emb_inps= torch.cat([start_batch, self.emb_layer(teacher_inps)], dim= 1)
            real_lens+= 1
            real_lens= real_lens.clamp(0, self.max_seq_len)
            real_lens_sorted, idx = real_lens.sort(0, descending=True)
            emb_inps_sorted = emb_inps[idx]
            h_0= torch.zeros((self.num_layers * self.num_directions, img_rep.size(0), self.output_size))
            packed_seq_x= pack_padded_sequence(emb_inps_sorted, real_lens_sorted, batch_first= self.batch_first)
            packed_out, packed_h_t= self.gru_layer(packed_seq_x, h_0)
            unpacked_out, _= pad_packed_sequence(packed_out, batch_first= self.batch_first, 
                                                 total_length= targets.size()[1])
            _, orig_idx = idx.sort(0)
            out = unpacked_out[orig_idx]
            final_out= self.vocab_project(out)
            return final_out
    
    def inference(self, img_rep):
        inp= img_rep.unsqueeze(1)
        state= torch.zeros((self.num_layers * self.num_directions, inp.size(0), self.output_size))
        word_ind_list= []
        for step in range(self.max_seq_len):
            output, state= self.gru_layer(inp, state)
            logits= self.vocab_project(output)
            m= torch.distributions.categorical.Categorical(logits= logits)
            word_inds= m.sample()
            word_ind_list.append(word_inds)
            inp= self.emb_layer(word_inds)
        output_inds= torch.cat(word_ind_list, dim=1)
        assert output_inds.size(1) == self.max_seq_len, "Incorrect sequence generated length"
        return output_inds

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
    def __init__(self, model, img_feature_size, word_emb_size):
        super(Encoder, self).__init__()
        modules= list(model.children())[:-1]      # delete the last fc layer.
        self.model= nn.Sequential(*modules)
        self.linear = nn.Linear(model.fc.in_features, word_emb_size)
    
    def forward(self, x):
#        pdb.set_trace()
        with torch.no_grad():
            x= self.model(x)
#        x= fn(self.model, x)
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

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1) # "flatten" the C * H * W values into a single vector per image



class DecoderfromFeatures(nn.Module):
    def __init__(self, img_feature_size, emb_mat, vocab_size, word_vec_size, num_layers, output_size, word_to_idx,
                 dropout= 0.5, bidirectional= False, trainable_embeddings= False, batch_first= True,
                 max_seq_len=20):
        super(DecoderfromFeatures, self).__init__()
        self.batch_first= batch_first
        self.max_seq_len= max_seq_len
        self.num_layers= num_layers
        self.bidirectional= bidirectional
        self.num_directions= 1 if bidirectional == False else 2
        self.output_size= output_size
        self.flatten= Flatten()
        self.lin_project_from_img = nn.Linear(img_feature_size, word_vec_size)
        self.emb_layer= nn.Embedding(vocab_size, word_vec_size)
        self.emb_layer.weight= nn.Parameter(torch.from_numpy(emb_mat.astype(np.float32)), 
                                            requires_grad= trainable_embeddings)
        self.gru_layer= nn.GRU(input_size= word_vec_size, hidden_size= output_size, num_layers= num_layers,
                           dropout= dropout, batch_first= self.batch_first, bidirectional= bidirectional)
        self.vocab_project= nn.Linear(output_size, vocab_size)
#        self.start_id= torch.tensor(word_to_idx[SpecialTokens.START], dtype= torch.long)
    
    def forward(self, img_rep, targets, real_lens, is_train= True):
        pdb.set_trace()
        img_flat= self.flatten(img_rep)
        img_projection= self.lin_project_from_img(img_flat)
        start_batch= img_projection.unsqueeze(1)
        if is_train == True: #replace with self.training later
            teacher_inps= targets[:, :-1]
            emb_inps= torch.cat([start_batch, self.emb_layer(teacher_inps)], dim= 1)
            real_lens+= 1
            real_lens= real_lens.clamp(0, self.max_seq_len)
            real_lens_sorted, idx = real_lens.sort(0, descending=True)
            emb_inps_sorted = emb_inps[idx]
            h_0= torch.zeros((self.num_layers * self.num_directions, img_projection.size(0), self.output_size))
            packed_seq_x= pack_padded_sequence(emb_inps_sorted, real_lens_sorted, batch_first= self.batch_first)
            packed_out, packed_h_t= self.gru_layer(packed_seq_x, h_0)
            unpacked_out, _= pad_packed_sequence(packed_out, batch_first= self.batch_first, 
                                                 total_length= targets.size()[1])
            _, orig_idx = idx.sort(0)
            out = unpacked_out[orig_idx]
            final_out= self.vocab_project(out)
            return final_out
    
    def inference(self, img_rep):
        inp= img_rep.unsqueeze(1)
        state= torch.zeros((self.num_layers * self.num_directions, inp.size(0), self.output_size))
        word_ind_list= []
        for step in range(self.max_seq_len):
            output, state= self.gru_layer(inp, state)
            logits= self.vocab_project(output)
            m= torch.distributions.categorical.Categorical(logits= logits)
            word_inds= m.sample()
            word_ind_list.append(word_inds)
            inp= self.emb_layer(word_inds)
        output_inds= torch.cat(word_ind_list, dim=1)
        assert output_inds.size(1) == self.max_seq_len, "Incorrect sequence generated length"
        return output_inds



class DecoderfromClassLogits(nn.Module):
    def __init__(self, emb_mat, vocab_size, word_vec_size, num_layers, output_size, word_to_idx,
                 dropout= 0.5, bidirectional= False, trainable_embeddings= False, batch_first= True,
                 max_seq_len=20):
        super(DecoderfromClassLogits, self).__init__()
        self.batch_first= batch_first
        self.max_seq_len= max_seq_len
        self.num_layers= num_layers
        self.bidirectional= bidirectional
        self.num_directions= 1 if bidirectional == False else 2
        self.output_size= output_size
        self.emb_layer= nn.Embedding(vocab_size, word_vec_size)
        self.emb_layer.weight= nn.Parameter(torch.from_numpy(emb_mat.astype(np.float32)), 
                                            requires_grad= trainable_embeddings)
        self.gru_layer= nn.GRU(input_size= word_vec_size, hidden_size= output_size, num_layers= num_layers,
                           dropout= dropout, batch_first= self.batch_first, bidirectional= bidirectional)
        self.vocab_project= nn.Linear(output_size, vocab_size)
    
    def forward(self, img_rep, targets, real_lens, is_train= True):
#        pdb.set_trace()
        start_batch= img_rep.unsqueeze(1)
        if is_train == True: #replace with self.training later
            teacher_inps= targets[:, :-1]
            emb_inps= torch.cat([start_batch, self.emb_layer(teacher_inps)], dim= 1)
            real_lens+= 1
            real_lens= real_lens.clamp(0, self.max_seq_len)
            real_lens_sorted, idx = real_lens.sort(0, descending=True)
            emb_inps_sorted = emb_inps[idx]
            h_0= torch.zeros((self.num_layers * self.num_directions, img_rep.size(0), self.output_size))
            packed_seq_x= pack_padded_sequence(emb_inps_sorted, real_lens_sorted, batch_first= self.batch_first)
            packed_out, packed_h_t= self.gru_layer(packed_seq_x, h_0)
            unpacked_out, _= pad_packed_sequence(packed_out, batch_first= self.batch_first, 
                                                 total_length= targets.size()[1])
            _, orig_idx = idx.sort(0)
            out = unpacked_out[orig_idx]
            final_out= self.vocab_project(out)
            return final_out
    
    def inference(self, img_rep):
        inp= img_rep.unsqueeze(1)
        state= torch.zeros((self.num_layers * self.num_directions, inp.size(0), self.output_size))
        word_ind_list= []
        for step in range(self.max_seq_len):
            output, state= self.gru_layer(inp, state)
            logits= self.vocab_project(output)
            m= torch.distributions.categorical.Categorical(logits= logits)
            word_inds= m.sample()
            word_ind_list.append(word_inds)
            inp= self.emb_layer(word_inds)
        output_inds= torch.cat(word_ind_list, dim=1)
        assert output_inds.size(1) == self.max_seq_len, "Incorrect sequence generated length"
        return output_inds
