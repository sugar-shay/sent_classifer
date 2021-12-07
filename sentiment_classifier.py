# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 17:19:37 2021

@author: Shadow
"""

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, ndcg_score

import torch
import torch.nn as nn 
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping



from transformers import AdamW
from transformers import AutoConfig, AutoModelForSequenceClassification


# =============================================================================
# We used Pytorch Lighting for rapid prototyping 
# ---> encapuslates the training loop
#----- We are able to use basically the same model as in our original testing except we change the loss function and one line of code in the forward() pass
# =============================================================================
class Lit_SequenceClassification(pl.LightningModule):
    
    def __init__(self, model_checkpoint, save_fp = 'best_model'):
        super(Lit_SequenceClassification, self).__init__()
        self.initialize_encoder(model_checkpoint)
        self.softmax = nn.Softmax(dim=-1)
        self.criterion = nn.CrossEntropyLoss()
        
        self.training_stats = {'train_losses':[],
                               'val_losses':[]}
        
        self.save_fp = save_fp
    
    def initialize_encoder(self, model_checkpoint):
        config = AutoConfig.from_pretrained(model_checkpoint, num_labels=3)
        self.encoder = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, config=config)
        
    def save_model(self):
        #torch.save(self.state_dict(), self.save_fp)
        self.encoder.save_pretrained(self.save_fp)
    
    '''
    def load_model(self, fp):
        self.load_state_dict(torch.load(fp))
    '''

    def forward(self, input_ids, attention_mask):
        
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        #logits = self.softmax(outputs.logits)
        
        
        #we return the outputs so that we can also evaluate the model using classification metrics if we so wish
        return outputs.logits
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=5e-6)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        
        logits = self.forward(input_ids= batch['input_ids'], attention_mask=batch['attention_mask'])
        #loss = self.criterion(logits, batch['label'])
        loss = torch.nn.functional.cross_entropy(logits, batch['label'])
        return {'loss':loss, 'train_loss':loss}
    
    def training_epoch_end(self, losses):
    
        avg_loss = (torch.tensor([x['loss'] for x in losses])).mean()
        self.training_stats['train_losses'].append(avg_loss.detach().cpu())
        print('Train Loss: ', avg_loss.detach().cpu().numpy())
        self.log('train_loss', avg_loss)
       
    def validation_step(self, batch, batch_idx):

        logits = self.forward(input_ids= batch['input_ids'], attention_mask=batch['attention_mask'])
        loss = torch.nn.functional.cross_entropy(logits, batch['label'])
    
        #loss = self.criterion(logits, batch['label'])

        return {'val_loss':loss}
  
    def validation_epoch_end(self, losses):
        avg_loss = (torch.tensor([x['val_loss'] for x in losses])).mean()        
        avg_loss_cpu = avg_loss.detach().cpu().numpy()
        if len(self.training_stats['val_losses']) == 0 or avg_loss_cpu<np.min(self.training_stats['val_losses']):
            self.save_model()
            
        self.training_stats['val_losses'].append(avg_loss_cpu)
        print('Val Loss: ', avg_loss_cpu)
        self.log('val_loss', avg_loss)
        
   
# =============================================================================
# Training function to train the model
# ---we use early stopping and monitor the validation loss
# =============================================================================
def train_LitModel(model, train_dataset, val_dataset, epochs, batch_size, patience = 3, num_gpu=1):
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle = False)
    
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=patience, verbose=False, mode="min")

    trainer = pl.Trainer(gpus=num_gpu, max_epochs = epochs, callbacks= [early_stop_callback])
    trainer.fit(model, train_dataloader, val_dataloader)
    
    return model


# =============================================================================
# This is the evaluation function for the Learn-to-Rank model
# ---returns NDCG@2 and NDCG@4
# ---returns classification metrics using the softmax logits from the encoder
# =============================================================================
def model_testing(model, test_dataset):
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model = model.to(device)
    
    test_dataloader = DataLoader(test_dataset, batch_size=32)
    
    preds, ground_truths = [], []
    
    model.eval()
    for idx, batch in enumerate(test_dataloader):
        
        seq = (batch['input_ids']).to(device)
        mask = (batch['attention_mask']).to(device)

        label = batch['label']
        ground_truths.extend(label)
        
        logits = model(input_ids=seq, attention_mask=mask)
        logits = logits.detach().cpu().numpy()
        preds.extend(np.argmax(logits, axis = -1))
    
    
    return preds, ground_truths

# =============================================================================
# This is the evaluation function for the Learn-to-Rank model
# ---returns NDCG@2 and NDCG@4
# ---returns classification metrics using the softmax logits from the encoder
# =============================================================================
def model_prediction(model, test_dataset):
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model = model.to(device)
    
    test_dataloader = DataLoader(test_dataset, batch_size=32)
    
    preds = []
    
    model.eval()
    for idx, batch in enumerate(test_dataloader):
        
        seq = (batch['input_ids']).to(device)
        mask = (batch['attention_mask']).to(device)

        
        logits = model(input_ids=seq, attention_mask=mask)
        logits = logits.detach().cpu().numpy()
        preds.extend(np.argmax(logits, axis = -1))
    
    
    return preds