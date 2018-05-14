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

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm

torch.manual_seed(1)

logger = logging.getLogger("Training")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class Decoder(nn.Module):
    def __init__(self, emb_mat, vocab_size, word_vec_size, num_layers, output_size, word_to_idx,
                 dropout= 0.5, bidirectional= False, trainable_embeddings= False):
        super(Decoder, self).__init__()
        self.emb_layer= nn.Embedding(vocab_size, word_vec_size)
        self.emb_layer= nn.Parameter(emb_mat, requires_grad= trainable_embeddings)
        self.gru_layer= nn.GRU(input_size= word_vec_size, hidden_size= output_size, num_layers= num_layers,
                           dropout= dropout, batch_first= False, bidirectional= bidirectional)
        self.vocab_project= nn.Linear(output_size, vocab_size)
        self.start_id= torch.tensor(word_to_idx[SpecialTokens.START], dtype= torch.long)
    
    def forward(self, img_rep, targets, real_lens, is_train= True):
        start_batch= self.start_id.repeat(targets.size()[0], 1)
        if is_train == True: #replace with self.training later
            teacher_inps= torch.cat([start_batch, targets[:, :-1]], dim= 1)
            emb_inps= self.emb_layer(teacher_inps)
            real_lens+= real_lens
            x= pack_padded_sequence(emb_inps, real_lens)
            h0= img_rep
            packed_out, packed_h_t= self.gru_layer(x, h0)
            outs, _= pad_packed_sequence(packed_out)
            
        else:
            raise NotImplementedError("Will implement inference mode")

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
    
    def forward(self, x):
        x= fn(self.model, x)
        x= x.view(x.size()[0], -1)
        x= self.linear(x)


class ImageCaption(nn.Module):
    def __init__(self, encoder, decoder):
        self.encoder= encoder
        self.decoder= decoder
    
    def forward(self, img, caption, real_lens):
        x= self.encoder(img)
        predictions= self.decoder(x, caption, real_lens)
        return predictions


if __name__ == "__main__":
    vocab_size, word_vec_size, num_layers, output_size= 10000, 300, 1, 512
#    emb_mat= np.random.randn(vocab_size, word_vec_size).astype(np.float32)
    token_dict, caption_dset= make_caption_word_dict()
    random_img, _= caption_dset[np.random.randint(1000)]
    #To get the linear proj size for encoder
    resnet18 = models.resnet18(pretrained=True)
    random_img= fn(resnet18, random_img.unsqueeze(0))
    img_feature_size= np.product(list(random_img.size()[1:]))
    #Get word vectors
    emb_matrix, word_to_idx= loadWordVectors(token_dict)
    my_dset= MyCOCODset(caption_dset, token_dict, 20, word_to_idx[SpecialTokens.PAD])
    #Construct encoder and decoder graphs
    encoder= Encoder(resnet18, img_feature_size, output_size)
    decoder= Decoder(emb_matrix, len(emb_matrix), word_vec_size, num_layers, output_size, word_to_idx)
    encoder.eval() #encoder is inference only
    #Combine into main model
    img_capt_model= ImageCaption(encoder, decoder)
    #Initialize dataloader: make separate ones for training and validation datasets once downloaded
    dataloader= DataLoader(my_dset, batch_size= 32,
                           shuffle=True, num_workers=4)
    
    num_epochs= 10 #Move into argparse
    best_score= 0.0 #Check against validation score after every epoch (or few steps)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(img_capt_model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        for i_batch, sample_batch in enumerate(dataloader):
            img_capt_model.zero_grad()
            img_batch, targets_batch, rlen_batch= sample_batch
            predictions= img_capt_model(img_batch, targets_batch, rlen_batch)
            loss= loss_function(predictions, targets_batch)
            loss.backward()
            optimizer.step()
        
# =============================================================================
#         Do validation here and save model if beats validation
# =============================================================================

