# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 09:05:04 2021

@author: Shadow
"""

import torch 
import transformers
from transformers import AutoTokenizer
from datasets import load_dataset

from sklearn.metrics import classification_report
import pickle

from tokenizer import *
from sentiment_classifier import *
def main():
    
    #model_checkpoint = 'ProsusAI/finbert'
    #model_checkpoint = 'bert-base-uncased'
    #model_checkpoint = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    model_checkpoint = "facebook/muppet-roberta-large"
    finetune_dataset = 'financial_phrasebank'
    
    #label 2 correspnds to positive sentiment 
    #label 1 is neutral 
    #label 0 is negative 
    train_data = load_dataset(finetune_dataset, 'sentences_75agree', split='train[:70%]')
    val_data = load_dataset(finetune_dataset, 'sentences_75agree', split='train[70%:85%]')
    test_data = load_dataset(finetune_dataset, 'sentences_75agree', split='train[85%:]')
    
    
    #need to find the average length of the sequences
    total_avg = sum( map(len, list(train_data['sentence'])) ) / len(train_data['sentence'])
    print('Avg. sentence length: ', total_avg)
    
    max_length = 192
    
    tokenizer = SentTokenizer(model_checkpoint, max_length)
    
    
    train_dataset = tokenizer.encode_data(train_data)
    val_dataset = tokenizer.encode_data(val_data)
    test_dataset = tokenizer.encode_data(test_data)
    

    model = Lit_SequenceClassification(model_checkpoint)
    
    preds, ground_truths = model_testing(model, test_dataset)
    
    cr = classification_report(y_true=ground_truths, y_pred = preds, output_dict = False)
    
    print('Untrained model CR: ', cr)
    
    
    model = train_LitModel(model, train_dataset, val_dataset, epochs=15, batch_size=16, patience = 3, num_gpu=1)
    
    '''
    #saving the training stats
    with open('train_stats.pkl', 'wb') as f:
        pickle.dump(model.training_stats, f)
    
    model = Lit_SequenceClassification(model_checkpoint = model_checkpoint,
                     save_fp='best_model.pt')
    
    model.load_state_dict(torch.load('best_model.pt'))
    '''
    preds, ground_truths = model_testing(model, test_dataset)
    
    cr = classification_report(y_true=ground_truths, y_pred = preds, output_dict = False)
    
    print()
    print('Trained model CR: ', cr)


if __name__ == "__main__":
    main()
    



        
    


