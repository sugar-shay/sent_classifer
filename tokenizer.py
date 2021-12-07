# -*- coding: utf-8 -*-

import torch
from transformers import AutoTokenizer

class SentTokenizer():
    
    def __init__(self, model_checkpoint, max_length, reddit_eval=False):
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
        self.max_len = max_length
        self.reddit_eval = reddit_eval
        
    def encode_data(self, data):
        
        sentences = data['sentence']
        deployment = False
        if 'label' not in data.columns:
            deployment = True
        
        if self.reddit_eval == False:
            encodings = self.tokenizer(sentences, is_split_into_words=False, return_offsets_mapping=False, max_length = self.max_len, padding=True, truncation=True)
            if deployment == False:
                dataset = Sentiment_Dataset(encodings, data['label'])
            else:
                 dataset = Sentiment_Dataset(encodings)
                 
        else:
            encodings = self.tokenizer(sentences.tolist(), is_split_into_words=False, return_offsets_mapping=False, max_length = self.max_len, padding=True, truncation=True)
            if deployment == False:
                dataset = Sentiment_Dataset(encodings, data['label'].tolist())
            else:
                 dataset = Sentiment_Dataset(encodings)
            
        return dataset
    

class Sentiment_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, ground_truths = None):
        self.encodings = encodings
        self.ground_truths = ground_truths
    
    def __getitem__(self, idx):
        item = {key: torch.LongTensor(val[idx]) for key, val in self.encodings.items()}
        if self.ground_truths is not None:
            item['label'] = torch.as_tensor(self.ground_truths[idx])
        return item
    
    def __len__(self):
        return len(self.encodings['input_ids'])

        