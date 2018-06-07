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
import pandas as pd
import torchvision.models as models
import torchvision.datasets as dset
import torchvision.transforms as transforms

import re
import datetime
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm

from ipdb import slaunch_ipdb_on_exception
import pdb
from saved_data import SavedData
import dill as pickle
from utils import (MyCOCODset, make_caption_word_dict, CocoDatasetDev,
                   CocoCaptions_Cust, CocoCaptions_Features, CocoCaptions_Features2)
from model import (Encoder, Decoder, fn, DecoderfromClassLogits, 
                   DecoderfromFeatures, AttentionDecoderClassLogits)
from image_captioning.build_vocab import SpecialTokens

import json
from pprint import pprint
from config import args
from scores import get_scores_im, get_scores

import logging
logger = logging.getLogger("Training")
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(levelname)s %(asctime)s : %(message)s', level=logging.INFO)

log_output= 'logging_' + args.model_name + args.feature_mode + str(args.is_attention).lower()
handler = logging.FileHandler(log_output)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(levelname)s %(asctime)s : %(message)s'))
logging.getLogger().addHandler(handler)

from collections import defaultdict

#Writing the hyperparameters to log file
for k, v in vars(args).items():
    logging.info("{}: {}".format(k, v))

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
#    try:
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, img_ids = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
#    except RuntimeError:
#        pdb.set_trace()
    return images, targets, torch.tensor(lengths).long(), img_ids


def get_dataloader(token_dict, mode= 'train', feature_shape= None, 
                   filename= None, vocab= None):
    if mode == 'train':
        data_transform= None
        caption_dset = CocoCaptions_Features2(root= filename, 
                                         annFile= args.train_caption_path,
                                         transform= data_transform,
                                         feature_shape= feature_shape,
                                         vocab= vocab)
        #Initialize dataloader: make separate ones for training and validation datasets once downloaded
        dataloader= DataLoader(caption_dset, batch_size= args.batch_size,
                               shuffle=True, num_workers= args.num_workers,
                               collate_fn=collate_fn)
        return dataloader, caption_dset
    elif mode == 'val':
        data_transform= transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), 
                                 (0.229, 0.224, 0.225))])
        caption_dset = CocoDatasetDev(root= filename, 
                                     annFile= args.val_caption_path,
                                     transform= data_transform,
                                     vocab= vocab)
        #Initialize dataloader: make separate ones for training and validation datasets once downloaded
        dataloader= DataLoader(caption_dset, batch_size= args.batch_size,
                               shuffle=False, num_workers= 0,
                               collate_fn=collate_fn)
        return dataloader, caption_dset
    else:
        raise NotImplementedError("Only validation and train")


def get_trainable_params(params):
    tr_params= []
    for param in params:
        if param.requires_grad == True:
            tr_params.append(param)
    return tr_params

def save_weights(decoder, epoch, step):
    model_path= (args.model_path_base + '/' + args.feature_mode + '/' +
                datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            )
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    else:
        pass
    #Later make this to save only after running eval and if better than best_score
    torch.save(decoder.state_dict(), os.path.join(
        model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, step+1)))
    return model_path


def save_best_weights(decoder, epoch, step):
    model_path= (args.model_path_base + '/' + args.feature_mode + '/' + args.model_name + '/' +
                datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            )
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    else:
        pass
    #Later make this to save only after running eval and if better than best_score
    torch.save(decoder.state_dict(), os.path.join(
        model_path, 'best_decoder-{}-{}.ckpt'.format(epoch+1, step+1)))
    return model_path

def save_to_json(val_gen_words_dict, metrics= None):
    temp_list= []
    for k, v in val_gen_words_dict.items():
        temp_dict= {}
        temp_dict["image_id"]= int(k)
        temp_dict["caption"]= str(v)
        temp_list.append(temp_dict)
    
    resFile='./evaluate/results/captions_val2017_results_' + \
                args.model_name + '_' + args.feature_mode + '_' + str(args.is_attention).lower() + '.json'
    with open(resFile, 'w') as out:
        json.dump(temp_list, out)
    
    scores= get_scores(resFile= resFile)
    if metrics is None:
        cum_score= np.mean([scores[k] for k in scores.keys()])
    else:
        cum_score= np.mean([scores[k] for k in scores.keys() if k in metrics])
    
    return cum_score


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
    
    return cum_score, scores

def get_model_wo_last_n_layers(model, n= 1):
    if n == 0:
        return model
    layers= []
    for name, layer in model._modules.items():
        layers.append(layer)
    return_model= torch.nn.Sequential(*layers[:-n])
    return return_model


def init_model(model_name, feature_mode):
    #download pre-trained models
    import torchvision.models as models
    resnet18 = models.resnet18(pretrained=True)
    resnet34 = models.resnet34(pretrained=True)
    resnet50 = models.resnet50(pretrained=True)
    resnet101 = models.resnet101(pretrained=True)
    resnet152 = models.resnet152(pretrained=True)
    alexnet = models.alexnet(pretrained=True)
    squeezenet = models.squeezenet1_1(pretrained=True)
    vgg16 = models.vgg16(pretrained=True)
    densenet = models.densenet161(pretrained=True)
    inception = models.inception_v3(pretrained=True)
    
    model_dict= {'resnet18': resnet18,
                'resnet34': resnet34,
                'resnet50': resnet50,
                'resnet101': resnet101,
                'resnet152': resnet152,
                'alexnet': alexnet,
                'squeezenet': squeezenet,
                'vgg16': vgg16,
                'densenet': densenet,
                'inception': inception,
                }
    
    model= model_dict[model_name]
    
    BASE_DIRECTORY= '/home/hanozbhathena/project/data/img_features/' + model_name
    
    if feature_mode == 'features':
        FILE_BASE= '_img_features.dat'
        model= get_model_wo_last_n_layers(model, n= 1)
    elif feature_mode == 'classes':
        FILE_BASE= '_img_distr.pickle'
        model= get_model_wo_last_n_layers(model, n= 0)
    
    train_features_file= os.path.join(BASE_DIRECTORY, 'train_' + args.model_name + FILE_BASE)
#    val_features_file= os.path.join(BASE_DIRECTORY, 'val_' + args.model_name + FILE_BASE)
    val_features_file= '/home/hanozbhathena/project/data/resizeval2017'
    return model, train_features_file, val_features_file



def get_feature_flat_dim(model):
    data_transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.ToTensor()
        ])
    train_caption_dset = CocoCaptions_Cust(root= '/home/hanozbhathena/project/data/resizetrain2017', 
                             annFile= '/home/hanozbhathena/project/data/annotations/captions_train2017.json',
                             transform= data_transform)
    val_caption_dset = CocoCaptions_Cust(root= '/home/hanozbhathena/project/data/resizeval2017', 
                             annFile= '/home/hanozbhathena/project/data/annotations/captions_val2017.json',
                             transform= data_transform)
    test= model(train_caption_dset[0][0].unsqueeze(0).to('cpu')).detach().cpu().numpy()
    train_shape= (len(train_caption_dset), *test.shape[1:])
    val_shape= (len(val_caption_dset), *test.shape[1:])
    return train_shape, val_shape


def evaluate(val_dataloader, encoder, decoder, idx_to_word, device, epoch, best_validation_score):
#    pdb.set_trace()
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        logging.info("Running on Validation set")
        val_gen_inds= []
        idx_list= []
        for i_batch, (image_batch, targets_batch, rlen_batch, idx_batch) in enumerate(val_dataloader):
            image_batch, targets_batch, rlen_batch= (image_batch.to(device), 
                                                     targets_batch.to(device), 
                                                     rlen_batch.to(device))
            img_features= encoder(image_batch)
            predictions_idx= decoder.inference(img_features)
            val_gen_inds.append(predictions_idx.cpu().numpy())
            idx_list.append(np.array(idx_batch))
            # Print log info
            if i_batch % args.log_step == 0:
                logging.info('Epoch [{}/{}], Dev Step {}'.format(epoch, args.num_epochs, i_batch))
    
#    pdb.set_trace()
    val_gen_inds= np.concatenate(val_gen_inds, axis=0)
    idx_concat= np.concatenate(idx_list, axis= 0)
    
    val_gen_inds_dict= dict(zip(idx_concat, val_gen_inds))
    val_gen_words_dict= {}
    for key, sent in val_gen_inds_dict.items():
        temp= []
        for ind in sent:
            word= idx_to_word[ind]
            if word == SpecialTokens().END or word == SpecialTokens().PAD:
                break
            temp.append(word)
        sentence= ' '.join(temp)
        val_gen_words_dict[key]= re.sub(SpecialTokens().START, "", sentence)
    validation_score, all_scores= evaluate_captions(val_gen_words_dict)
    if validation_score > best_validation_score:
        save_to_json(val_gen_words_dict)
    logging.info("Epoch {} validation captions saved to".format(epoch, args.output_json))
    decoder.train()
    return validation_score, all_scores


def write_scores(model_path, model_scores_dict, ddict= None):
#    pdb.set_trace()
    with open(os.path.join(model_path, 'val_scores.txt'), 'a') as fout:
        for k, v in model_scores_dict.items():
            fout.write(str(k) + " : " + str(v) + "\n")
        fout.write('\n\n\n')
    
    if ddict is not None:
        loss_csv= os.path.join(model_path, 'loss_record.csv')
        loss_df= pd.DataFrame(ddict['loss'])
        loss_df.to_csv(loss_csv)
        
        with open(os.path.join(model_path, 'full_record.pkl'), 'wb') as fout:
            pickle.dump(ddict, fout)


def count_params(model, parameters= None):
    if parameters is None:
        parameters= model.parameters()
    param_size_list= []
    for param in parameters:
        param_size_list.append(np.product(param.size()))
    print("Total Model parameters = {}".format(np.sum(param_size_list)))
    param_size_list= []
    for param in parameters:
        if param.requires_grad:
            param_size_list.append(np.product(param.size()))
    print("Total Model trainable parameters = {}".format(np.sum(param_size_list)))


def main():
#    pdb.set_trace()
    with open(args.save_data_fname, 'rb') as input_:
        vocab= pickle.load(input_)
        emb_matrix, word_to_idx, idx_to_word= vocab.word_embeddings, vocab.word2idx, vocab.idx2word
    #set device
    if args.use_cuda == True:
        if torch.cuda.is_available() == False:
            logging.info("GPU not available; defaulting to CPU")
            devicename= 'cpu'
        else:
            logging.info("Using GPU")
            devicename= 'cuda'
    else:
        devicename= 'cpu'
    device = torch.device(devicename)
    cnn_model, train_features_file, val_features_file= init_model(args.model_name, args.feature_mode)
    
    train_shape, val_shape= get_feature_flat_dim(cnn_model.to('cpu'))
    img_feature_size= np.product(train_shape[1:])
    #Construct decoder graph
    if args.feature_mode == 'features':
        decoder= DecoderfromFeatures(img_feature_size, emb_matrix, len(emb_matrix), 
                                     args.embed_size, args.num_layers, 
                                     args.hidden_size, word_to_idx).to(device)
    else:
        if args.is_attention:
            decoder= AttentionDecoderClassLogits(img_feature_size, emb_matrix, len(emb_matrix), 
                                         args.embed_size, args.num_layers, 
                                         args.hidden_size, word_to_idx).to(device)
        else:
            decoder= DecoderfromClassLogits(img_feature_size, emb_matrix, len(emb_matrix), 
                                         args.embed_size, args.num_layers, 
                                         args.hidden_size, word_to_idx).to(device)
    
#        pdb.set_trace()
    train_dataloader, my_dset_train= get_dataloader(word_to_idx, 'train', train_shape, train_features_file,
                                                    vocab= vocab)
    val_dataloader, my_dset_val= get_dataloader(word_to_idx, 'val', val_shape, val_features_file,
                                                vocab= vocab)
    
    vocab_size= len(emb_matrix)
    loss_function = nn.CrossEntropyLoss(reduce= False)
    all_trainable_params= get_trainable_params(list(decoder.parameters()))
    optimizer = optim.Adam(all_trainable_params, lr= args.learning_rate)
    test_steps= 10
    best_val_score= 0.0 #Check against validation score after every epoch (or few steps)
    all_scores_dict= None
    cnn_model= cnn_model.to(device)
    count_params(decoder)
#    pdb.set_trace()
    record_dict= defaultdict(list)
    for epoch in range(args.num_epochs):
        #Training
        decoder.train()
        for i_batch, (img_features, targets_batch, rlen_batch, idx_batch) in enumerate(train_dataloader):
            img_features, targets_batch= (img_features.to(device), 
                                          targets_batch.to(device))
            rlen_batch= rlen_batch.to(device)
            predictions= decoder(img_features, targets_batch, rlen_batch)
            targets_batch= targets_batch.view(-1)
            predictions= predictions.view(-1, vocab_size)
            loss_matrix= loss_function(predictions, targets_batch)
            mask= 1 - targets_batch.eq(word_to_idx[SpecialTokens().PAD])
            loss= torch.mean(loss_matrix.masked_select(mask))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            record_dict['loss'].append(loss.detach().cpu().numpy())
#            if i_batch >= test_steps:
#                break
            
            # Print log info
            if i_batch % args.log_step == 0:
                logging.info('Epoch [{}/{}], Train Step {}, Train Loss: {:.4f}, Train Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i_batch, loss.item(), np.exp(loss.item()))) 
                
            # Save the model checkpoints
            if (i_batch+1) % args.save_step == 0:
                #Later make this to save only after running eval and if better than best_score
                model_path= save_weights(decoder, epoch, i_batch)
#                if all_scores_dict is not None:
#                    write_scores(model_path, all_scores_dict)
            
            if args.intra_epoch_eval == True:
                if i_batch > 0 and i_batch % args.eval_frequency == 0:
                    curr_validation_score, all_scores_dict= evaluate(val_dataloader, encoder= cnn_model, 
                                                     decoder= decoder, idx_to_word= idx_to_word, 
                                                     device= device, epoch= epoch,
                                                     best_validation_score= best_val_score)
                    if curr_validation_score > best_val_score:
                        #Save weights at end of the epoch; best weights can be ensembled
                        best_model_path= save_best_weights(decoder, epoch, i_batch)
                        best_val_score= curr_validation_score
                        record_dict['best_val_score'].append(best_val_score)
                        record_dict['all_scores_dict'].append(all_scores_dict)
                        write_scores(best_model_path, all_scores_dict, record_dict)
        
        curr_validation_score, all_scores_dict= evaluate(val_dataloader, encoder= cnn_model, 
                                                 decoder= decoder, idx_to_word= idx_to_word, 
                                                 device= device, epoch= epoch,
                                                 best_validation_score= best_val_score)
        
        if curr_validation_score > best_val_score:
            #Save weights at end of the epoch; best weights can be ensembled
            best_model_path= save_best_weights(decoder, epoch, i_batch)
            best_val_score= curr_validation_score
            record_dict['best_val_score'].append(best_val_score)
            record_dict['all_scores_dict'].append(all_scores_dict)
            write_scores(best_model_path, all_scores_dict, record_dict)

if __name__ == "__main__":
    with slaunch_ipdb_on_exception():
        main()
