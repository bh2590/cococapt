#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 17:22:44 2018

@author: hanozbhathena
"""


import os
import numpy as np
import pickle
import pdb
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
from PIL import Image

logger = logging.getLogger("Data preprocessing")
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
from image_captioning.build_vocab import SpecialTokens


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
        # mask = [True] * max_length if len_sentence >= max_length else [True] * len(sentence) 
#        + [False] * (max_length - len_sentence)
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
    
    caption_dset = CocoCaptions_Cust(root= root, annFile= annFile,
                                     transform= data_transform)
    
    words_set= set()
    for i in tqdm(range(len(caption_dset))):
        _, caption, _= caption_dset[i]
        tokens= set(word_tokenize(' '.join(caption).lower()))
        words_set= words_set.union(tokens)
    
    word_list= list(words_set)
    word_to_idx= dict(zip(word_list, range(1, 1 + len(word_list))))
    idx_to_word= {v:k for k,v in word_to_idx.items()}
    assert 0 not in idx_to_word, "PAD token should be 0 index"
    return word_to_idx, idx_to_word, caption_dset

#def loadWordVectors(tokens, filepath="/Users/hanozbhathena/Documents/coco/data/glove.840B.300d.txt", dimensions=300):
#    """Read pretrained GloVe vectors"""
#    dic = {}
#    words= []
#    with open(filepath) as ifs:
#        for line in ifs:
#            line = line.strip()
#            if not line:
#                continue
#            row = line.split()
#            token = row[0]
#            if token not in tokens:
#                continue
#            try:
#                data = [float(x) for x in row[1:]]
#            except ValueError:
#                words.append(' '.join(row[:-300]))
#                continue
#            if len(data) != dimensions:
#                raise RuntimeError("wrong number of dimensions")
#            dic[token] = np.asarray(data)
#    logging.info("{} lines were not meeting split standards: {}".format(len(words), words))
#    dic[SpecialTokens.START]= np.random.uniform(low=-0.25, high=0.25, size=dimensions)
#    dic[SpecialTokens.END]= np.random.uniform(low=-0.25, high=0.25, size=dimensions)
#    dic[SpecialTokens.OOV]= np.random.uniform(low=-0.25, high=0.25, size=dimensions)
#    dic[SpecialTokens.PAD]= np.random.uniform(low=-0.25, high=0.25, size=dimensions)
#    
#    tokens[SpecialTokens.PAD]= 0
#    tokens[SpecialTokens.START]= len(tokens)
#    tokens[SpecialTokens.END]= len(tokens)
#    tokens[SpecialTokens.OOV]= len(tokens)
#    
#    tokens_inv= {v:k for k,v in tokens.items()}
#    
#    pdb.set_trace()
#    wordVectors = np.zeros((len(tokens), dimensions))
#    oov_count= 0
#    for word, ind in tokens.items():
#        try:
#            wordVectors[ind]= dic[word]
#        except KeyError:
#            oov_count+=1
#            wordVectors[ind]= dic[SpecialTokens.OOV]
#    logging.info("{} words from COCO were OOV for Glove".format(oov_count))
#    logging.info("word vectors loaded using" + filepath)
#    return wordVectors, tokens, tokens_inv

def clean_text(text):
    #Logic to clean text: review from Keras tutorial or some COCO captions tutorial online
    return text

def get_word_id(w, word_to_idx):
    try:
        return word_to_idx[w]
    except KeyError:
        return word_to_idx[SpecialTokens().OOV]


class MyCOCODset(Dataset):
    def __init__(self, coco_dset, word_to_idx, max_seq_len, pad_id):
        self.coco_dset= coco_dset
        self.word_to_idx= word_to_idx
        self.max_seq_len= max_seq_len
        self.pad_id= pad_id
    
    def __getitem__(self, ind):
        img, self.str_caption, img_idx= self.coco_dset[ind]
        self.str_caption= self.str_caption[np.random.randint(len(self.str_caption))].lower()
        ind_caption= [get_word_id(w, self.word_to_idx) for w in word_tokenize(clean_text(self.str_caption))]
        real_len= min(self.max_seq_len, len(ind_caption))
        ind_caption= ind_caption + [self.pad_id] * (self.max_seq_len - real_len)
        return img, torch.tensor(ind_caption[:self.max_seq_len], dtype= torch.long), real_len, img_idx
    
    def __len__(self):
        return len(self.coco_dset)


class CocoCaptions_Cust(Dataset):
    def __init__(self, root, annFile, transform=None, target_transform=None):
        from pycocotools.coco import COCO
        self.root = os.path.expanduser(root)
        self.coco = COCO(annFile)
        self.ids = list(self.coco.anns.keys())
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        coco = self.coco
        ann_id = self.ids[index]
        target = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, target, img_id

    def __len__(self):
        return len(self.ids)
#
#
#    coco = self.coco
#    img_id = self.ids[index]
#    ann_ids = coco.getAnnIds(imgIds=img_id)
#    anns = coco.loadAnns(ann_ids)
#    target = [ann['caption'] for ann in anns]
#    path = coco.loadImgs(img_id)[0]['file_name']
#    img = Image.open(os.path.join(self.root, path)).convert('RGB')
#
#
#    def __init__(self, root, annFile, transform=None, target_transform=None):
#        from pycocotools.coco import COCO
#        self.root = os.path.expanduser(root)
#        self.coco = COCO(annFile)
#        self.ids = list(self.coco.imgs.keys())
#        self.transform = transform
#        self.target_transform = target_transform


#class CocoCaptions_Cust(Dataset):
#    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
#    def __init__(self, root, json, vocab, transform=None):
#        """Set the path for images, captions and vocabulary wrapper.
#        
#        Args:
#            root: image directory.
#            json: coco annotation file path.
#            vocab: vocabulary wrapper.
#            transform: image transformer.
#        """
#        self.root = root
#        self.coco = COCO(json)
#        self.ids = list(self.coco.anns.keys())
#        self.vocab = vocab
#        self.transform = transform
#
#    def __getitem__(self, index):
#        """Returns one data pair (image and caption)."""
#        pdb.set_trace()
#        coco = self.coco
#        vocab = self.vocab
#        ann_id = self.ids[index]
#        caption = coco.anns[ann_id]['caption']
#        img_id = coco.anns[ann_id]['image_id']
#        path = coco.loadImgs(img_id)[0]['file_name']
#
#        image = Image.open(os.path.join(self.root, path)).convert('RGB')
#        if self.transform is not None:
#            image = self.transform(image)
#
#        # Convert caption (string) to word ids.
#        tokens = word_tokenize(str(caption).lower())
#        caption = []
#        caption.append(vocab('<start>'))
#        caption.extend([vocab(token) for token in tokens])
#        caption.append(vocab('<end>'))
#        target = torch.Tensor(caption)
#        return image, target
#
#    def __len__(self):
#        return len(self.ids)


class CocoCaptions_Features(Dataset):
    def __init__(self, root, annFile, feature_shape= None, transform=None, target_transform=None):
        from pycocotools.coco import COCO
        self.features_file = os.path.expanduser(root)
        self.coco = COCO(annFile)
        self.ids = list(self.coco.anns.keys())
        self.transform = transform
        self.target_transform = target_transform
        
        file_type= self.features_file.split('.')[-1]
        
        if file_type == 'pickle' or file_type == 'pkl':
            with open(self.features_file, 'rb') as fin:
                self.features= pickle.load(fin)
        elif file_type == 'dat':
            if feature_shape is None:
                raise ValueError("features shape must be provided")
            self.features= np.memmap(self.features_file, dtype= np.float32, mode= 'r', shape= feature_shape)
        else:
            raise ValueError("Incompatible features file type")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (feature, target). target is a list of captions for the image.
        """

        pdb.set_trace()
        coco = self.coco
        ann_id = self.ids[index]
        target = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
#        path = coco.loadImgs(img_id)[0]['file_name']
#
#        image = Image.open(os.path.join(self.root, path)).convert('RGB')
#        if self.transform is not None:
#            image = self.transform(image)
        
        feature = self.features[index]
        feature = torch.tensor(feature)
        if self.transform is not None:
            feature = self.transform(feature)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return feature, target, img_id

    def __len__(self):
        return len(self.ids)




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
