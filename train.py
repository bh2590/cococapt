#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 16:51:07 2018

@author: hanozbhathena
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np
import torchvision.models as models
import torchvision.datasets as dset
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm
import logging
logger = logging.getLogger("Training")
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

from ipdb import slaunch_ipdb_on_exception
import ipdb as pdb
from saved_data import SavedData
import pickle
from utils import SpecialTokens, MyCOCODset, make_caption_word_dict, loadWordVectors, CocoCaptions_Cust
from model import Encoder, Decoder, fn

import ast
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='./weights/' , help='path for saving trained models')
parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
parser.add_argument('--glove_path', type=str, default='/Users/hanozbhathena/Documents/coco/data/glove.840B.300d.txt', 
                    help='path for pretrained glove embeddings')

parser.add_argument('--train_image_dir', type=str, default='/Users/hanozbhathena/Documents/coco/data/val2017', 
                    help='directory for train images')
parser.add_argument('--val_image_dir', type=str, default='/Users/hanozbhathena/Documents/coco/data/val2017', 
                    help='directory for val images')

parser.add_argument('--train_caption_path', type=str, 
                    default='/Users/hanozbhathena/Documents/coco/data/annotations/captions_val2017.json', 
                    help='path for train annotation json file')
parser.add_argument('--val_caption_path', type=str, 
                    default='/Users/hanozbhathena/Documents/coco/data/annotations/captions_val2017.json', 
                    help='path for val annotation json file')

parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')

# Model parameters
parser.add_argument('--embed_size', type=int , default=300, help='dimension of word embedding vectors')
parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
parser.add_argument('--preprocess', type=ast.literal_eval , default=True, help='whether to preprocess from scratch')
parser.add_argument('--use_cuda', type=ast.literal_eval , default=False, help='whether to use GPU')
parser.add_argument('--save_data_fname', type=str , default='data.pickle', 
                    help='file to save/load embedding matrix and word_to_idx dict')
parser.add_argument('--max_seq_len', type=int , default=20, help='maximum unrolling length')

parser.add_argument('--num_epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=0.001)
args = parser.parse_args()


def get_dataloader(token_dict, mode= 'train'):
    if mode == 'train':
        data_transform = transforms.Compose([
                transforms.RandomSizedCrop(args.crop_size),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        caption_dset = CocoCaptions_Cust(root= args.train_image_dir, 
                                         annFile= args.train_caption_path,
                                         transform= data_transform)
        #Subclass COCO datset
        my_dset= MyCOCODset(caption_dset, token_dict, args.max_seq_len, word_to_idx[SpecialTokens.PAD])
        #Initialize dataloader: make separate ones for training and validation datasets once downloaded
        dataloader= DataLoader(my_dset, batch_size= args.batch_size,
                               shuffle=True, num_workers= args.num_workers)
        return dataloader
    elif mode == 'val':
        data_transform = transforms.Compose([
                transforms.RandomSizedCrop(args.crop_size),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        caption_dset = CocoCaptions_Cust(root= args.val_image_dir, 
                                         annFile= args.val_caption_path,
                                         transform= data_transform)
        #Subclass COCO datset
        my_dset= MyCOCODset(caption_dset, token_dict, args.max_seq_len, word_to_idx[SpecialTokens.PAD])
        #Initialize dataloader: make separate ones for training and validation datasets once downloaded
        dataloader= DataLoader(my_dset, batch_size= args.batch_size,
                               shuffle=False, num_workers= args.num_workers)
        return dataloader
    else:
        raise NotImplementedError("Only validation and train")

def get_trainable_params(params):
    tr_params= []
    for param in params:
        if param.requires_grad == True:
            tr_params.append(param)
    return tr_params


if __name__ == "__main__":
    with slaunch_ipdb_on_exception():
        pdb.set_trace(context= 5)
        if args.preprocess == True:
            token_dict, _, caption_dset= make_caption_word_dict(args.train_image_dir, args.train_caption_path)
            #Get word vectors
            emb_matrix, word_to_idx, idx_to_word= loadWordVectors(token_dict)
            #Save for future use
            obj= SavedData(emb_matrix, word_to_idx, idx_to_word)
            with open(args.save_data_fname, 'wb') as output:
                pickle.dump(obj, output)
        else:
            with open(args.save_data_fname, 'rb') as input_:
                obj= pickle.load(input_)
                emb_matrix, word_to_idx, idx_to_word= obj.emb_mat, obj.word_to_idx, obj.idx_to_word
                data_transform = transforms.Compose([
                        transforms.RandomSizedCrop(args.crop_size),
                        #transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
                caption_dset = CocoCaptions_Cust(root= args.train_image_dir, 
                                                 annFile= args.train_caption_path,
                                                 transform= data_transform)
        random_img, _, _= caption_dset[np.random.randint(1000)]
        #To get the linear proj size for encoder
        resnet18 = models.resnet18(pretrained=True)
        random_img= fn(resnet18, random_img.unsqueeze(0))
        img_feature_size= np.product(list(random_img.size()[1:]))
        #set device
        device = torch.device("cuda" if args.use_cuda else "cpu")
        #Construct encoder and decoder graphs
        encoder= Encoder(resnet18, img_feature_size, args.embed_size).to(device)
        decoder= Decoder(emb_matrix, len(emb_matrix), args.embed_size, args.num_layers, 
                         args.hidden_size, word_to_idx).to(device)
        
        train_dataloader= get_dataloader(word_to_idx, 'train')
        val_dataloader= get_dataloader(word_to_idx, 'val')
        
        vocab_size= len(emb_matrix)
        best_score= 0.0 #Check against validation score after every epoch (or few steps)
        loss_function = nn.CrossEntropyLoss(reduce= False)
        all_params= get_trainable_params(list(decoder.parameters())) + get_trainable_params(list(encoder.linear.parameters()))
        optimizer = optim.Adam(all_params, lr= args.learning_rate)
        pdb.set_trace()
        for epoch in range(args.num_epochs):
            #Training
            encoder.linear.train()
            decoder.train()
            for i_batch, (img_batch, targets_batch, rlen_batch, idx_batch) in enumerate(train_dataloader):
                img_batch, targets_batch, rlen_batch= (img_batch.to(device), 
                                                       targets_batch.to(device), 
                                                       rlen_batch.to(device))
                img_features= encoder(img_batch)
                predictions= decoder(img_features, targets_batch, rlen_batch)
                targets_batch= targets_batch.view(-1)
                predictions= predictions.view(-1, vocab_size)
                loss_matrix= loss_function(predictions, targets_batch)
                mask= 1 - targets_batch.eq(word_to_idx[SpecialTokens.PAD])
                loss= torch.mean(loss_matrix.masked_select(mask))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                # Print log info
                if i_batch % args.log_step == 0:
                    print('Epoch [{}/{}], Step {}, Loss: {:.4f}, Perplexity: {:5.4f}'
                          .format(epoch, args.num_epochs, i_batch, loss.item(), np.exp(loss.item()))) 
                    
                # Save the model checkpoints
                if (i_batch+1) % args.save_step == 0:
                    #Later make this to save only after running eval and if better than best_score
                    torch.save(decoder.state_dict(), os.path.join(
                        args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i_batch+1)))
                    torch.save(encoder.state_dict(), os.path.join(
                        args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i_batch+1)))
                
#                break
            
            with torch.no_grad():
                pdb.set_trace()
                val_gen_inds= []
                for i_batch, (img_batch, targets_batch, rlen_batch, idx_batch) in enumerate(val_dataloader):
                    img_batch, targets_batch, rlen_batch= (img_batch.to(device), 
                                                           targets_batch.to(device), 
                                                           rlen_batch.to(device))
                    img_features= encoder(img_batch)
                    predictions_idx= decoder.inference(img_features)
#                    predictions_idx= predictions_idx.numpy()
                    val_gen_inds.append(predictions_idx)
#                    if i_batch >= 2:
#                        break
                pdb.set_trace()
                val_gen_inds= torch.cat(val_gen_inds, dim=0)
                val_gen_inds= val_gen_inds.numpy()
                val_gen_words= []
                for sent in val_gen_inds:
                    temp= []
                    for ind in sent:
                        word= idx_to_word[ind]
                        if word == SpecialTokens.END or word == SpecialTokens.PAD:
                            break
                        temp.append(word)
                    val_gen_words.append(' '.join(temp))
                #Send val_gen_words to coco validation

