# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 11:52:41 2021

@author: Shadow
"""

import torch 
import transformers
from transformers import AutoTokenizer
from datasets import load_dataset

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

from tokenizer import *
from sentiment_classifier import *

#model_checkpoint = 'ProsusAI/finbert'
#model_checkpoint = 'bert-base-uncased'
#model_checkpoint = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
model_checkpoint = "facebook/muppet-roberta-large"
testing_dataset = 'Reddit_Data.csv'

#label 2 correspnds to positive sentiment 
#label 1 is neutral 
#label 0 is negative 

#train_data = load_dataset(finetune_dataset, 'sentences_75agree', split='train[:70%]')
#val_data = load_dataset(finetune_dataset, 'sentences_75agree', split='train[70%:85%]')
reddit_data = pd.read_csv(testing_dataset)
reddit_data.columns = ['sentence', 'label']

reddit_data['sentence'] = reddit_data['sentence'].astype(str)
reddit_data['label'] = reddit_data['label'].astype(int)

reddit_data = reddit_data.sample(frac=1)

train_data, val_data = reddit_data.iloc[:400, :], reddit_data.iloc[400:, :]



#need to find the average length of the sequences
total_avg = sum( map(len, list(train_data['sentence'])) ) / len(val_data['sentence'])
print('Avg. sentence length: ', total_avg)


max_length = 192
tokenizer = SentTokenizer(model_checkpoint, max_length, reddit_eval=True)


train_dataset = tokenizer.encode_data(train_data)
val_dataset = tokenizer.encode_data(val_data)
#test_dataset = tokenizer.encode_data(test_data)


model = Lit_SequenceClassification(model_checkpoint)

preds, ground_truths = model_testing(model, val_dataset)

cr = classification_report(y_true=ground_truths, y_pred = preds, output_dict = False)

print()
print('Reddit Data Base Report: ')
print(cr)

model = train_LitModel(model, train_dataset, val_dataset, epochs=15, batch_size=8, patience = 3, num_gpu=1)
    
 
preds, ground_truths = model_testing(model, val_dataset)

cr = classification_report(y_true=ground_truths, y_pred = preds, output_dict = False)

print()
print('Reddit Data Finetune Report: ')
print(cr)


   
'''

if __name__ == "__main__":
    main()
    
'''