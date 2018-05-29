#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 23:04:42 2018

@author: hanozbhathena
"""

import ast
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='weights' , help='path for saving trained models')
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
parser.add_argument('--output_json', type=str, 
                    default='val_generated_capts.json', 
                    help='val generated captions filename')


parser.add_argument('--num_epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=0.001)
args = parser.parse_args()