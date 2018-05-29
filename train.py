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
logging.basicConfig(format='%(levelname)s %(asctime)s : %(message)s', level=logging.INFO)

from ipdb import slaunch_ipdb_on_exception
import ipdb as pdb
from saved_data import SavedData
import pickle
from utils import SpecialTokens, MyCOCODset, make_caption_word_dict, loadWordVectors, CocoCaptions_Cust
from model import Encoder, Decoder, fn

import json
from pprint import pprint
from config import args
from scores import get_scores_im


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
        return dataloader, my_dset
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
        return dataloader, my_dset
    else:
        raise NotImplementedError("Only validation and train")

def get_trainable_params(params):
    tr_params= []
    for param in params:
        if param.requires_grad == True:
            tr_params.append(param)
    return tr_params

def save_weights(encoder, decoder, epoch, step):
    #Later make this to save only after running eval and if better than best_score
    torch.save(decoder.state_dict(), os.path.join(
        args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, step+1)))
    torch.save(encoder.state_dict(), os.path.join(
        args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, step+1)))

def save_to_json(val_gen_words_dict):
    temp_list= []
    for k, v in val_gen_words_dict.items():
        temp_dict= {}
        temp_dict["image_id"]= int(k)
        temp_dict["caption"]= str(v)
        temp_list.append(temp_dict)
    
    with open(args.output_json, 'w') as out:
        json.dump(temp_list, out)


def evaluate_captions(val_gen_words_dict, metrics= None):
    temp_list= []
    for k, v in val_gen_words_dict.items():
        temp_dict= {}
        temp_dict["image_id"]= int(k)
        temp_dict["caption"]= str(v)
        temp_list.append(temp_dict)
    
    scores= get_scores_im(result_list= temp_list)
    
    if metrics is None:
        cum_score= np.mean([scores[k] for k in scores.keys()])
    else:
        cum_score= np.mean([scores[k] for k in scores.keys() if k in metrics])
    
    return cum_score


if __name__ == "__main__":
    with slaunch_ipdb_on_exception():
        pdb.set_trace(context= 5)
        with open(args.save_data_fname, 'rb') as input_:
            vocab= pickle.load(input_)
            emb_matrix, word_to_idx, idx_to_word= vocab.word_embeddings, vocab.word2idx, vocab.idx2word
        data_transform = transforms.Compose([
                transforms.RandomSizedCrop(args.crop_size),
                transforms.RandomHorizontalFlip(),
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
        
        train_dataloader, my_dset_train= get_dataloader(word_to_idx, 'train')
        val_dataloader, my_dset_val= get_dataloader(word_to_idx, 'val')
        
        vocab_size= len(emb_matrix)
        best_score= 0.0 #Check against validation score after every epoch (or few steps)
        loss_function = nn.CrossEntropyLoss(reduce= False)
        all_params= get_trainable_params(list(decoder.parameters())) + get_trainable_params(
                list(encoder.linear.parameters()))
        optimizer = optim.Adam(all_params, lr= args.learning_rate)
#        pdb.set_trace()
        for epoch in range(args.num_epochs):
            #Training
            pdb.set_trace()
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
                    logging.info('Epoch [{}/{}], Train Step {}, Train Loss: {:.4f}, Train Perplexity: {:5.4f}'
                          .format(epoch, args.num_epochs, i_batch, loss.item(), np.exp(loss.item()))) 
                    
                # Save the model checkpoints
                if (i_batch+1) % args.save_step == 0:
                    #Later make this to save only after running eval and if better than best_score
                    save_weights(encoder, decoder, epoch, i_batch)
            
            #Save weights at end of the epoch
            save_weights(encoder, decoder, epoch, i_batch)
            
            with torch.no_grad():
                logging.info("Running on Validation set")
#                pdb.set_trace()
                val_gen_inds= []
                idx_list= []
                for i_batch, (img_batch, targets_batch, rlen_batch, idx_batch) in enumerate(val_dataloader):
                    img_batch, targets_batch, rlen_batch= (img_batch.to(device), 
                                                           targets_batch.to(device), 
                                                           rlen_batch.to(device))
                    img_features= encoder(img_batch)
                    predictions_idx= decoder.inference(img_features)
                    val_gen_inds.append(predictions_idx)
                    idx_list.append(idx_batch)
                    # Print log info
                    if i_batch % args.log_step == 0:
                        logging.info('Epoch [{}/{}], Dev Step {}'
                              .format(epoch, args.num_epochs, i_batch))
                        
#                pdb.set_trace()
                val_gen_inds= torch.cat(val_gen_inds, dim=0)
                idx_concat= torch.cat(idx_list, dim= 0).numpy()
                val_gen_inds= val_gen_inds.numpy()
                
                val_gen_inds_dict= dict(zip(idx_concat, val_gen_inds))
                val_gen_words_dict= {}
                for key, sent in val_gen_inds_dict.items():
                    temp= []
                    for ind in sent:
                        word= idx_to_word[ind]
                        if word == SpecialTokens.END or word == SpecialTokens.PAD:
                            break
                        temp.append(word)
                    val_gen_words_dict[key]= ' '.join(temp)
                save_to_json(val_gen_words_dict)
                logging.info("Epoch {} validation captions saved to".format(epoch, args.output_json))
                #Send val_gen_words_dict to coco validation

