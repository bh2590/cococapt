#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 21:10:01 2018

@author: hanozbhathena
"""

class SavedData(object):
    def __init__(self, emb_mat, word_to_idx):
        self.emb_mat= emb_mat
        self.word_to_idx= word_to_idx