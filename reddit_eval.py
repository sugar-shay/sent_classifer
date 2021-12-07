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
training_dataset = 'Reddit_Data.csv'
testing_dataset = 'Reddit_Data_Test.csv'


#label 2 correspnds to positive sentiment 
#label 1 is neutral 
#label 0 is negative 

#train_data = load_dataset(finetune_dataset, 'sentences_75agree', split='train[:70%]')
#val_data = load_dataset(finetune_dataset, 'sentences_75agree', split='train[70%:85%]')
reddit_train_data, reddit_test_data = pd.read_csv(training_dataset), pd.read_csv(testing_dataset)
reddit_train_data.columns, reddit_test_data.columns = ['sentence', 'label'], ['sentence', 'label'] 

reddit_train_data['sentence'], reddit_test_data['sentence'] = reddit_train_data['sentence'].astype(str), reddit_test_data['sentence'].astype(str)
reddit_train_data['label'], reddit_test_data['label'] = reddit_train_data['label'].astype(int), reddit_test_data['label'].astype(int)

reddit_train_data = reddit_train_data.sample(frac=1)

train_data, val_data = reddit_train_data.iloc[:400, :], reddit_train_data.iloc[400:, :]



#need to find the average length of the sequences
total_avg = sum( map(len, list(train_data['sentence'])) ) / len(train_data['sentence'])
print('Avg. sentence length: ', total_avg)


max_length = 192
tokenizer = SentTokenizer(model_checkpoint, max_length, reddit_eval=True)


train_dataset = tokenizer.encode_data(train_data)
val_dataset = tokenizer.encode_data(val_data)
test_dataset = tokenizer.encode_data(reddit_test_data)


model = Lit_SequenceClassification(model_checkpoint, save_fp = 'reddit_sentiment_classifier')

preds, ground_truths = model_testing(model, test_dataset)

cr = classification_report(y_true=ground_truths, y_pred = preds, output_dict = False)

print()
print('Reddit Data Base Report: ')
print(cr)

model = train_LitModel(model, train_dataset, val_dataset, epochs=15, batch_size=8, patience = 3, num_gpu=1)
    
 
preds, ground_truths = model_testing(model, test_dataset)

cr = classification_report(y_true=ground_truths, y_pred = preds, output_dict = False)

print()
print('Reddit Data Finetune Report: ')
print(cr)

model = Lit_SequenceClassification('reddit_sentiment_classifier')
#model.load_model('reddit_sentiment_classifier.pt')

preds, ground_truths = model_testing(model, test_dataset)

cr = classification_report(y_true=ground_truths, y_pred = preds, output_dict = False)

print()
print('Reddit Data Finetune Report w/ Save: ')
print(cr)



   
'''

if __name__ == "__main__":
    main()
    
'''