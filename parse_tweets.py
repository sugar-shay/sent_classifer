# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 17:11:56 2021

@author: Shadow
"""

import pandas as pd

from sentiment_classifier import *
from tokenizer import *

from sklearn.metrics import classification_report
from collections import Counter

test_data = pd.read_pickle('raw_tweet_data.pkl')
text = test_data['text'].tolist()

tokenizer_checkpoint = "facebook/muppet-roberta-large"
model_checkpoint = 'reddit_sentiment_classifier'
sanity_check_dataset = 'Reddit_Data_Test.csv'


text = pd.DataFrame(text)
text.columns = ['sentence']
text['sentence'] = text['sentence'].astype(str)

sanity_check_data = pd.read_csv(sanity_check_dataset)
sanity_check_data.columns = ['sentence', 'label']

sanity_check_data['sentence'] = sanity_check_data['sentence'].astype(str)
sanity_check_data['label'] =  sanity_check_data['label'].astype(int)


max_length = 192
tokenizer = SentTokenizer(tokenizer_checkpoint, max_length, reddit_eval=True)

sanity_check_dataset = tokenizer.encode_data(sanity_check_data)
test_dataset = tokenizer.encode_data(text)

model = Lit_SequenceClassification('reddit_sentiment_classifier')

sanity_check_preds, ground_truths = model_testing(model, sanity_check_dataset)

cr = classification_report(y_true=ground_truths, y_pred = sanity_check_preds, output_dict = False)

print()
print('Sanity Check on Reddit Test Data: ')
print(cr)

test_preds = model_prediction(model, test_dataset)

processed_twitter_data = pd.DataFrame(data = {'sentiment': test_preds, 'date': test_data['date'].tolist()})
