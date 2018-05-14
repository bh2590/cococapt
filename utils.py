#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 17:22:44 2018

@author: hanozbhathena
"""


import os
import numpy as np
import pickle
import ipdb as pdb
from saved_data import SavedData
import re
import multiprocessing as mp
import logging

from collections import Counter
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm

logger = logging.getLogger("Data preprocessing")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class SpecialTokens:
    END = '<EOS>'
    OOV = "<OOV>"
    START = "<S>"
    PAD = '<PAD>'


def build_dict(words, max_words=None, offset=0):
    cnt = Counter(words)
    if max_words:
        words = cnt.most_common(max_words)
    else:
        words = cnt.most_common()
    return {word: offset + i for i, (word, _) in enumerate(words)}


def sentence2id(sentence, tok2id):
    if_oov = tok2id.get("<OOV>")
    sentence_ = [tok2id.get(word, if_oov) for word in sentence]
    return sentence_


def corpora2id(data, tok2id):
    return np.array([sentence2id(sentence, tok2id) for sentence in data])


def pad_sequences(data, max_length, start_skip=0):
    ret = []
    real_lens = []
    null_token = '<NULL>'
    for sentence in data:
        sentence = sentence[start_skip:]
        len_sentence = len(sentence)
        if len_sentence >= max_length:
            new_sentence = sentence[:max_length]
        else:
            for i in range(0, max_length - len_sentence):
                sentence = np.append(sentence, null_token)
            new_sentence = sentence
        real_lens.append(min(len_sentence, max_length))
        # print "len(sentence)    ", len(sentence), "start_skip ", start_skip
        # mask = [True] * max_length if len_sentence >= max_length else [True] * len(sentence) + [False] * (max_length - len_sentence)
        ret.append(new_sentence)
    return np.array(ret), np.array(real_lens)


def make_caption_word_dict(root = '/Users/hanozbhathena/Documents/coco/data/val2017',
                           annFile = '/Users/hanozbhathena/Documents/coco/data/annotations/captions_val2017.json'):
    data_transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                     std=[0.229, 0.224, 0.225])
        ])
    
    caption_dset = dset.CocoCaptions(root= root, annFile= annFile,
                                     transform= data_transform)
    
    words_set= set()
    for i in tqdm(range(len(caption_dset))):
        _, caption= caption_dset[i]
        tokens= set(word_tokenize(' '.join(caption).lower()))
        words_set= words_set.union(tokens)
    
    word_list= list(words_set)
    word_to_idx= dict(zip(word_list, range(len(word_list))))
    return word_to_idx, caption_dset

def loadWordVectors(tokens, filepath="/Users/hanozbhathena/Documents/coco/data/glove.840B.300d.txt", dimensions=300):
    """Read pretrained GloVe vectors"""
    wordVectors = np.zeros((len(tokens), dimensions))
    dic = {}
    words= []
    with open(filepath) as ifs:
        for line in ifs:
            line = line.strip()
            if not line:
                continue
            row = line.split()
            token = row[0]
            if token not in tokens:
                continue
            try:
                data = [float(x) for x in row[1:]]
            except ValueError:
                words.append(' '.join(row[:-300]))
                continue
            if len(data) != dimensions:
                raise RuntimeError("wrong number of dimensions")
            dic[token] = np.asarray(data)
    logging.info("{} lines were not meeting split standards: {}".format(len(words), words))
    dic[SpecialTokens.START]= np.random.uniform(low=-0.25, high=0.25, size=dimensions)
    dic[SpecialTokens.END]= np.random.uniform(low=-0.25, high=0.25, size=dimensions)
    dic[SpecialTokens.OOV]= np.random.uniform(low=-0.25, high=0.25, size=dimensions)
    dic[SpecialTokens.PAD]= np.random.uniform(low=-0.25, high=0.25, size=dimensions)
    
    tokens[SpecialTokens.START]= len(tokens)
    tokens[SpecialTokens.END]= len(tokens)
    tokens[SpecialTokens.OOV]= len(tokens)
    tokens[SpecialTokens.PAD]= len(tokens)
    
    oov_count= 0
    for word, ind in tokens.items():
        try:
            wordVectors[ind]= dic[word]
        except KeyError:
            oov_count+=1
            wordVectors[ind]= dic[SpecialTokens.OOV]
    logging.info("{} words from COCO were OOV for Glove".format(oov_count))
    logging.info("word vectors loaded using" + filepath)
    return wordVectors, tokens

def clean_text(text):
    #Logic to clean text: review from Keras tutorial or some COCO captions tutorial online
    return text


class MyCOCODset(Dataset):
    def __init__(self, coco_dset, word_to_idx, max_seq_len, pad_id):
        self.coco_dset= coco_dset
        self.word_to_idx= word_to_idx
        self.max_seq_len= max_seq_len
        self.pad_id= pad_id
    
    def __getitem__(self, ind):
        img, str_caption= self.coco_dset[ind]
        str_caption= str_caption[np.random.randint(len(str_caption))].lower()
        ind_caption= [self.word_to_idx[w] for w in word_tokenize(clean_text(str_caption))]
        real_len= len(ind_caption)
        ind_caption= ind_caption + [self.pad_id] * (self.max_seq_len - real_len)
        return img, torch.tensor(ind_caption, dtype= torch.long), real_len
    
    def __len__(self):
        return len(self.coco_dset)


def main(filename= 'saved_data.pickle'):
    tokens= make_caption_word_dict()
    emb_mat, token_dict= loadWordVectors(tokens)
    obj= SavedData(emb_mat, token_dict)
    with open(filename, 'wb') as output:
        pickle.dump(obj, output)
    logger.info("Data Saved!")


def fake_data(n, seqlen, vsize):
    from random_words import RandomWords
    rw = RandomWords()
    pass


if __name__ == "__main__":
    main()
