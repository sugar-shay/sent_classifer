# -*- coding: utf-8 -*-

import torch
from transformers import AutoTokenizer

class SentTokenizer():
    
    def __init__(self, model_checkpoint, max_length):
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
        self.mex_len = max_length
        
    def encode_data(self, data):
        
        sentences = data['sentence']
        encodings = self.tokenizer(sentences, is_split_into_words=False, return_offsets_mapping=False, max_length = self.max_len, padding=True, truncation=True)
        dataset = Sentiment_Dataset(encodings, data['label'])
        
        return dataset
    

class Sentiment_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, ground_truths):
        self.encodings = encodings
        self.ground_truths = ground_truths
    
    def __getitem__(self, idx):
        item = {key: torch.LongTensor(val[idx]) for key, val in self.encodings.items()}
        if self.ground_truths is not None:
            item['label'] = torch.as_tensor(self.ground_truth[idx])
        return item
    
    def __len__(self):
        return len(self.encodings['input_ids'])

        