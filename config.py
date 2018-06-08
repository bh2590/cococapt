#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 23:04:42 2018

@author: hanozbhathena
"""

import ast
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path_base', type=str, default='/home/hanozbhathena/project/weights' , 
                    help='path for saving trained models')
parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
parser.add_argument('--glove_path', type=str, default='/home/hanozbhathena/project/data/glove.840B.300d.txt', 
                    help='path for pretrained glove embeddings')

parser.add_argument('--train_image_dir', type=str, default='/home/hanozbhathena/project/data/img_features', 
                    help='directory for train images')
parser.add_argument('--val_image_dir', type=str, default='/home/hanozbhathena/project/data/img_features', 
                    help='directory for val images')

parser.add_argument('--train_caption_path', type=str, 
                    default='/home/hanozbhathena/project/data/annotations/captions_train2017.json', 
                    help='path for train annotation json file')
parser.add_argument('--val_caption_path', type=str, 
                    default='/home/hanozbhathena/project/data/annotations/captions_val2017.json', 
                    help='path for val annotation json file')

parser.add_argument('--log_step', type=int , default=100, help='step size for prining log info')
parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')

# Model parameters
parser.add_argument('--model_name', type=str, default='resnet18', 
                    help='model name')
parser.add_argument('--feature_mode', type=str, default='classes', 
                    help='either one of features or classes')
parser.add_argument('--embed_size', type=int , default=300, help='dimension of word embedding vectors')
parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
parser.add_argument('--num_layers', type=int , default=2, help='number of layers in lstm')
parser.add_argument('--preprocess', type=ast.literal_eval , default=True, help='whether to preprocess from scratch')
parser.add_argument('--use_cuda', type=ast.literal_eval, default=True, help='whether to use GPU')
parser.add_argument('--save_data_fname', type=str , default='/home/hanozbhathena/project/data/vocab.pkl', 
                    help='file to save/load embedding matrix and word_to_idx dict')
parser.add_argument('--max_seq_len', type=int , default=20, help='maximum unrolling length')
parser.add_argument('--output_json', type=str, 
                    default='val_generated_capts.json', 
                    help='val generated captions filename')
parser.add_argument('--topk_classes', type=int, default=5)
parser.add_argument('--pretrained_embeddings', type=ast.literal_eval, default=False, 
                    help='whether to use pre trained embeddings or train embeddings from scratch')
parser.add_argument('--intra_epoch_eval', type=ast.literal_eval , default=True, 
                    help='whether to run eval on validation set before epoch over')
parser.add_argument('--eval_frequency', type=int, default=2000,
                    help='if intra_epoch_eval is true then the number of steps between running evaluation')
parser.add_argument('--is_attention', type=ast.literal_eval , default=True, 
                    help='whether to use attention or not')
parser.add_argument('--class_attention', type=ast.literal_eval , default=True, 
                    help='whether to run class attention or not')
parser.add_argument('--decoder_attention', type=ast.literal_eval , default=True, 
                    help='whether to run intra-decoder attention or not')

#Training parameters
parser.add_argument('--num_epochs', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=0.001)


args = parser.parse_args()