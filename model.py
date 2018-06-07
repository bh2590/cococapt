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
from utils import SpecialTokens
import numpy as np
import torchvision.models as models
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm
from config import args
#torch.manual_seed(1)

from ipdb import slaunch_ipdb_on_exception
import pdb

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
        self.start_id= torch.tensor(word_to_idx[SpecialTokens().START], dtype= torch.long)
    
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
    def __init__(self):
        super(Flatten, self).__init__()
    
    def forward(self, x):
        N = x.size(0)
        return x.view(N, -1)



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
        if args.pretrained_embeddings == True:
            self.emb_layer.weight= nn.Parameter(torch.from_numpy(emb_mat.astype(np.float32)), 
                                                requires_grad= trainable_embeddings)
        self.gru_layer= nn.GRU(input_size= word_vec_size, hidden_size= output_size, num_layers= num_layers,
                           dropout= dropout, batch_first= self.batch_first, bidirectional= bidirectional)
        self.vocab_project= nn.Linear(output_size, vocab_size)
        self.bn = nn.BatchNorm1d(word_vec_size, momentum=0.01)
    
    def forward(self, img_rep, targets, real_lens):
        img_flat= self.flatten(img_rep) # (b, C, H, W) 
        img_projection= self.lin_project_from_img(img_flat) # (b, D)
        img_projection= self.bn(img_projection) # (b, D)
        start_batch= img_projection.unsqueeze(1) # (b, 1, D)
        teacher_inps= targets[:, :-1] # (b, L-1) ints/word ids
        emb_inps= torch.cat([start_batch, self.emb_layer(teacher_inps)], dim= 1)
#        real_lens= real_lens.clamp(0, self.max_seq_len)
        real_lens_sorted, idx = real_lens.sort(0, descending=True)
        emb_inps_sorted = emb_inps[idx]
#        h_0= torch.zeros((self.num_layers * self.num_directions, img_projection.size(0), self.output_size))
        packed_seq_x= pack_padded_sequence(emb_inps_sorted, real_lens_sorted, batch_first= self.batch_first)
        packed_out, packed_h_t= self.gru_layer(packed_seq_x)
        unpacked_out, _= pad_packed_sequence(packed_out, batch_first= self.batch_first, 
                                             total_length= targets.size()[1])
        _, orig_idx = idx.sort(0)
        out = unpacked_out[orig_idx]
        final_out= self.vocab_project(out)
        return final_out
    
    def inference_sample(self, img_rep):
        img_flat= self.flatten(img_rep)
        img_projection= self.lin_project_from_img(img_flat)
        inp= img_projection.unsqueeze(1)
#        state= torch.zeros((self.num_layers * self.num_directions, inp.size(0), self.output_size))
        word_ind_list= []
        for step in range(self.max_seq_len):
            if step == 0:
                output, state= self.gru_layer(inp)
            else:
                output, state= self.gru_layer(inp, state)
            logits= self.vocab_project(output)
            m= torch.distributions.categorical.Categorical(logits= logits)
            word_inds= m.sample()
            word_ind_list.append(word_inds)
            inp= self.emb_layer(word_inds)
        output_inds= torch.cat(word_ind_list, dim=1)
#        assert output_inds.size(1) == self.max_seq_len, "Incorrect sequence generated length"
        return output_inds


class DecoderfromClassLogits(nn.Module):
    def __init__(self, img_feature_size, emb_mat, vocab_size, word_vec_size, num_layers, output_size, word_to_idx,
                 dropout= 0.5, bidirectional= False, trainable_embeddings= False, batch_first= True,
                 max_seq_len=20):
        super(DecoderfromClassLogits, self).__init__()
        self.batch_first= batch_first
        self.max_seq_len= max_seq_len
        self.num_layers= num_layers
        self.bidirectional= bidirectional
        self.num_directions= 1 if bidirectional == False else 2
        self.output_size= output_size
#        self.flatten= Flatten()
        self.topk_classes= args.topk_classes
        self.class_emb_layer= nn.Embedding(1000, word_vec_size)
        self.lin_project_from_img = nn.Linear(word_vec_size, word_vec_size)
        self.emb_layer= nn.Embedding(vocab_size, word_vec_size)
        if args.pretrained_embeddings == True:
            self.emb_layer.weight= nn.Parameter(torch.from_numpy(emb_mat.astype(np.float32)), 
                                                requires_grad= trainable_embeddings)
        self.gru_layer= nn.GRU(input_size= word_vec_size, hidden_size= output_size, num_layers= num_layers,
                           dropout= dropout, batch_first= self.batch_first, bidirectional= bidirectional)
        self.vocab_project= nn.Linear(output_size, vocab_size)
        self.bn= nn.BatchNorm1d(word_vec_size, momentum=0.01)
    
    def forward(self, img_rep, targets, real_lens, is_train= True):
        _, argmax= torch.topk(img_rep, self.topk_classes, dim=1) # argmax: (b, topk_classes)
        class_emb= self.class_emb_layer(argmax) # class_emb: (b, topk, word_vec_size)
        class_emb= torch.mean(class_emb, 1) # class_emb: (b, word_vec_size)
        img_projection= self.lin_project_from_img(class_emb) # class_emb: (b, word_vec_size)
        img_projection= self.bn(img_projection) # (b, word_vec_size)
        start_batch= img_projection.unsqueeze(1) # (b, 1, word_vec_size)
        teacher_inps= targets[:, :-1] # (b, L-1)
        emb_inps= torch.cat([start_batch, self.emb_layer(teacher_inps)], dim= 1) # (b, L-1, word_vec_size)
#        real_lens= real_lens.clamp(0, self.max_seq_len)
        real_lens_sorted, idx = real_lens.sort(0, descending=True)
        emb_inps_sorted = emb_inps[idx]
#        h_0= torch.zeros((self.num_layers * self.num_directions, img_projection.size(0), self.output_size))
        packed_seq_x= pack_padded_sequence(emb_inps_sorted, real_lens_sorted, batch_first= self.batch_first)
        packed_out, packed_h_t= self.gru_layer(packed_seq_x)
        unpacked_out, _= pad_packed_sequence(packed_out, batch_first= self.batch_first, 
                                             total_length= targets.size()[1])
        _, orig_idx = idx.sort(0)
        out = unpacked_out[orig_idx]
        final_out= self.vocab_project(out)
        return final_out
    
    def inference_sample(self, img_rep):
#        pdb.set_trace()
        _, argmax= torch.topk(img_rep, self.topk_classes, dim=1)
        class_emb= self.class_emb_layer(argmax)
        class_emb= torch.mean(class_emb, 1)
        img_projection= self.lin_project_from_img(class_emb)
        inp= img_projection.unsqueeze(1)
        word_ind_list= []
        for step in range(self.max_seq_len):
            if step == 0:
                output, state= self.gru_layer(inp)
            else:
                output, state= self.gru_layer(inp, state)
            logits= self.vocab_project(output)
            m= torch.distributions.categorical.Categorical(logits= logits)
            word_inds= m.sample()
            word_ind_list.append(word_inds)
            inp= self.emb_layer(word_inds)
        output_inds= torch.cat(word_ind_list, dim=1)
#        assert output_inds.size(1) == self.max_seq_len, "Incorrect sequence generated length"
        return output_inds
    
    def inference(self, img_rep, states=None): #greedy
        """Generate captions for given image features using greedy search."""
        _, argmax= torch.topk(img_rep, self.topk_classes, dim=1) # argmax: (b, topk_classes)
        class_emb= self.class_emb_layer(argmax) # class_emb: (b, topk, word_vec_size)
        class_emb= torch.mean(class_emb, 1) # class_emb: (b, word_vec_size)
        img_projection= self.lin_project_from_img(class_emb) # class_emb: (b, word_vec_size)
        img_projection= self.bn(img_projection) # (b, word_vec_size)
        inp= img_projection.unsqueeze(1) # (b, 1, word_vec_size)
        sampled_ids = []
        for step in range(self.max_seq_len):
            if step == 0:
                hiddens, states= self.gru_layer(inp)
            else:
                hiddens, states= self.gru_layer(inp, states)
            outputs = self.vocab_project(hiddens.squeeze(1))            # outputs:  (B, V)
            _, predicted = outputs.max(1)                               # predicted: (B)
            sampled_ids.append(predicted)
            inp = self.emb_layer(predicted)                       # inp: (B, D)
            inp = inp.unsqueeze(1)                                # inp: (B, 1, D)
        sampled_ids = torch.stack(sampled_ids, 1)                 # sampled_ids: (B, max_seq_length)
        return sampled_ids


class DecoderfromFeatures2(nn.Module):
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
    
    def forward(self, img_rep, targets, real_lens):
#        pdb.set_trace()
        img_flat= self.flatten(img_rep)
        img_projection= self.lin_project_from_img(img_flat)
        start_batch= img_projection.unsqueeze(1)
        teacher_inps= targets[:, :-1]
        emb_inps= torch.cat([start_batch, self.emb_layer(teacher_inps)], dim= 1)
        real_lens+= 1
        real_lens= real_lens.clamp(0, self.max_seq_len)
        real_lens_sorted, idx = real_lens.sort(0, descending=True)
        emb_inps_sorted = emb_inps[idx]
#        h_0= torch.zeros((self.num_layers * self.num_directions, img_projection.size(0), self.output_size))
        packed_seq_x= pack_padded_sequence(emb_inps_sorted, real_lens_sorted, batch_first= self.batch_first)
        packed_out, packed_h_t= self.gru_layer(packed_seq_x)
        unpacked_out, _= pad_packed_sequence(packed_out, batch_first= self.batch_first, 
                                             total_length= targets.size()[1])
        _, orig_idx = idx.sort(0)
        out = unpacked_out[orig_idx]
        final_out= self.vocab_project(out)
        return final_out
    
    def inference_sample(self, img_rep):
        img_flat= self.flatten(img_rep)
        img_projection= self.lin_project_from_img(img_flat)
        inp= img_projection.unsqueeze(1)
#        state= torch.zeros((self.num_layers * self.num_directions, inp.size(0), self.output_size))
        word_ind_list= []
        for step in range(self.max_seq_len):
            if step == 0:
                output, state= self.gru_layer(inp)
            else:
                output, state= self.gru_layer(inp, state)
            logits= self.vocab_project(output)
            m= torch.distributions.categorical.Categorical(logits= logits)
            word_inds= m.sample()
            word_ind_list.append(word_inds)
            inp= self.emb_layer(word_inds)
        output_inds= torch.cat(word_ind_list, dim=1)
        assert output_inds.size(1) == self.max_seq_len, "Incorrect sequence generated length"
        return output_inds


class AttentionLayer(nn.Module):
    def __init__(self, param_size):
        super(AttentionLayer, self).__init__()
        weight= torch.rand(param_size)
        self.weight= nn.Parameter(weight) # (hidden_size, hidden_size)
    
    def forward(self, ht, h_0_t):
        """
        ht: current output of GRU at step t: (batch_size, 1, hidden_size)
        h_0_t: all previous t-1 GRU outputs: (batch_size, t-1, hidden_size)
        """
        batch_size= ht.size(0)
        Wbatch= self.weight.repeat(batch_size, 1, 1) # (batch_size, hidden_size, hidden_size)
        bmm1= torch.bmm(ht, Wbatch) # (batch_size, 1, hidden_size)
        h_0_t_transpose= h_0_t.permute(0, 2, 1) # (batch_size, hidden_size, t-1)
        bmm2= torch.bmm(bmm1, h_0_t_transpose) # (batch_size, 1, t-1)
        bmm2= bmm2.squeeze(1) # (batch_size, t-1)
        alpha= torch.nn.functional.softmax(bmm2, dim=1) # (batch_size, t-1)
        alpha= alpha.unsqueeze(-1) # (batch_size, t-1, 1)
        temp= alpha * h_0_t # (batch_size, t-1, hidden_size)
        ct= torch.sum(temp, dim=1, keepdim=True) # (batch_size, 1, hidden_size)
        return ct


class AttentionDecoderClassLogits(nn.Module):
    def __init__(self, img_feature_size, emb_mat, vocab_size, word_vec_size, num_layers, output_size, word_to_idx,
                 dropout= 0.5, bidirectional= False, trainable_embeddings= False, batch_first= True,
                 max_seq_len=20):
        super(AttentionDecoderClassLogits, self).__init__()
        self.batch_first= batch_first
        self.max_seq_len= max_seq_len
        self.num_layers= num_layers
        self.bidirectional= bidirectional
        self.num_directions= 1 if bidirectional == False else 2
        self.output_size= output_size
#        self.flatten= Flatten()
        self.topk_classes= args.topk_classes
        self.class_emb_layer= nn.Embedding(1000, word_vec_size)
        self.lin_project_from_img = nn.Linear(word_vec_size, word_vec_size)
        self.emb_layer= nn.Embedding(vocab_size, word_vec_size)
        if args.pretrained_embeddings == True:
            self.emb_layer.weight= nn.Parameter(torch.from_numpy(emb_mat.astype(np.float32)), 
                                                requires_grad= trainable_embeddings)
        self.gru_layer= nn.GRU(input_size= word_vec_size, hidden_size= output_size, num_layers= num_layers,
                           dropout= dropout, batch_first= self.batch_first, bidirectional= bidirectional)
        self.bn= nn.BatchNorm1d(word_vec_size, momentum=0.01)
        self.attn_layer= AttentionLayer((output_size, output_size))
        self.vocab_project= nn.Linear(2 * output_size, vocab_size)
    
    def forward(self, img_rep, targets, real_lens, is_train= True):
        _, argmax= torch.topk(img_rep, self.topk_classes, dim=1) # argmax: (b, topk_classes)
        class_emb= self.class_emb_layer(argmax) # class_emb: (b, topk, word_vec_size)
        class_emb= torch.mean(class_emb, 1) # class_emb: (b, word_vec_size)
#        class_emb= class_emb.view(class_emb.size(0), -1)
        img_projection= self.lin_project_from_img(class_emb) # class_emb: (b, word_vec_size)
        img_projection= self.bn(img_projection) # (b, word_vec_size)
        start_batch= img_projection.unsqueeze(1) # (b, 1, word_vec_size)
        teacher_inps= targets[:, :-1] # (b, L-1)
        emb_inps= torch.cat([start_batch, self.emb_layer(teacher_inps)], dim= 1) # (b, L-1, word_vec_size)
#        real_lens= real_lens.clamp(0, self.max_seq_len)
        real_lens_sorted, idx = real_lens.sort(0, descending=True)
        emb_inps_sorted = emb_inps[idx]
        packed_seq_x= pack_padded_sequence(emb_inps_sorted, real_lens_sorted, batch_first= self.batch_first)
        packed_out, packed_h_t= self.gru_layer(packed_seq_x)
        unpacked_out, _= pad_packed_sequence(packed_out, batch_first= self.batch_first, 
                                             total_length= targets.size()[1])
        _, orig_idx = idx.sort(0)
        out = unpacked_out[orig_idx]
        num_steps= out.size(1)
        
        context_vector_list= []
        for t in range(num_steps):
            if t == 0:
                ct= torch.zeros_like(out[:, t, :].unsqueeze(1))
            else:
                ct= self.attn_layer(ht= out[:, t, :].unsqueeze(1), h_0_t= out[:, :t, :])
            context_vector_list.append(ct)
        
        context_vector= torch.cat(context_vector_list, dim= 1)
        context_out= torch.cat([out, context_vector], dim= 2)
        final_out= self.vocab_project(context_out)
        return final_out
    
    def inference(self, img_rep, states=None): #greedy
        """Generate captions for given image features using greedy search."""
        _, argmax= torch.topk(img_rep, self.topk_classes, dim=1) # argmax: (b, topk_classes)
        class_emb= self.class_emb_layer(argmax) # class_emb: (b, topk, word_vec_size)
        class_emb= torch.mean(class_emb, 1) # class_emb: (b, word_vec_size)
#        class_emb= class_emb.view(class_emb.size(0), -1)
        img_projection= self.lin_project_from_img(class_emb) # class_emb: (b, word_vec_size)
        img_projection= self.bn(img_projection) # (b, word_vec_size)
        inp= img_projection.unsqueeze(1) # (b, 1, word_vec_size)
        sampled_ids = []
        hidden_list= []
        for step in range(self.max_seq_len):
            if step == 0:
                hiddens, states= self.gru_layer(inp)
                context_vector= torch.zeros_like(hiddens)
            else:
                hiddens, states= self.gru_layer(inp, states)
                context_vector= self.attn_layer(ht= hiddens, h_0_t= torch.cat(hidden_list, dim= 1))
            hidden_list.append(hiddens)
            vocab_input= torch.cat([hiddens, context_vector], dim= 2)
            outputs = self.vocab_project(vocab_input.squeeze(1)) # outputs:  (B, V)
            _, predicted = outputs.max(1) # predicted: (B)
            sampled_ids.append(predicted)
            inp = self.emb_layer(predicted) # inp: (B, D)
            inp = inp.unsqueeze(1) # inp: (B, 1, D)
        sampled_ids = torch.stack(sampled_ids, 1) # sampled_ids: (B, max_seq_length)
        return sampled_ids



