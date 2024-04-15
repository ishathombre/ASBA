import torch
import torch.nn as nn
from transformers import DebertaModel, DebertaForSequenceClassification
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification
from transformers import TrainingArguments, Trainer, TrainerCallback
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score
#import evaluate
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


class RobertaForASBA(RobertaForSequenceClassification):
    def __init__(self, config, num_aspect_labels, num_polarization_labels):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.aspect_classifier = nn.Linear(config.hidden_size, num_aspect_labels)
        self.polarization_classifier = nn.Linear(config.hidden_size, num_polarization_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)
        #print(outputs)
        pooled_output = self.dropout(outputs[1])
        aspect_logits = self.aspect_classifier(pooled_output)
        polarization_logits = self.polarization_classifier(pooled_output)

        aspect_loss = None
        polarization_loss = None
        if labels is not None:
            aspect_loss = F.cross_entropy(aspect_logits, labels[:,0])
            polarization_loss = F.cross_entropy(polarization_logits, labels[:,1])
            total_loss = aspect_loss + polarization_loss
        else:
            total_loss = None

        return total_loss, aspect_loss, polarization_loss, aspect_logits, polarization_logits


tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
train_dataset, val_dataset= setup_data(rests, tokenizer, 2228, 2970)

num_aspect_labels = 4
num_polarization_labels = 5
model = RobertaForASBA.from_pretrained('FacebookAI/roberta-base', num_aspect_labels=num_aspect_labels, num_polarization_labels=num_polarization_labels)


def training_run(model, num_epochs, batch_size, train_dataset, val_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    training_aspect = []
    training_polarization = []
    validation_aspect = []
    validation_polarization = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        total_aspect_train_loss = 0.0
        total_pol_train_loss = 0.0

        for batch in train_dataloader:
            print("here")
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            total_loss, aspect_loss, polarization_loss, aspect_logits, polarization_logits = model(input_ids=input_ids, labels=labels)

            total_loss.backward()
            optimizer.step()

            total_train_loss += total_loss.item()
            total_aspect_train_loss += aspect_loss.item()
            total_pol_train_loss += polarization_loss.item()
            
        
            #print(aspect_loss, "aspect_loss")
            training_aspect.append(aspect_loss)
            #print(polarization_loss, "polarization_loss")
            training_polarization.append(polarization_loss)
        

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_aspect_train_loss = total_aspect_train_loss / len(train_dataloader)
        avg_pol_train_loss = total_pol_train_loss / len(train_dataloader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        total_aspect_val_loss = 0.0
        total_pol_val_loss = 0.0
    

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                total_loss, aspect_loss, polarization_loss, aspect_logits, polarization_logits = model(input_ids=input_ids, labels=labels)
                total_val_loss += total_loss.item()
                total_aspect_val_loss += aspect_loss.item()
                total_pol_val_loss += polarization_loss.item()
                
                #print(aspect_loss, "aspect_loss_validation")
                validation_aspect.append(aspect_loss)
                #print(polarization_loss, "polarization_loss_validation")
                validation_polarization.append(polarization_loss)

        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_aspect_val_loss = total_aspect_val_loss / len(train_dataloader)
        avg_pol_val_loss = total_pol_val_loss / len(train_dataloader)

        print(f'Epoch {epoch + 1}/{num_epochs}: Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')
        #print(f'Epoch {epoch + 1}/{num_epochs}: Training Loss Aspect: {avg_aspect_train_loss}, Validation Loss Aspect: {avg_aspect_val_loss}')
        #print(f'Epoch {epoch + 1}/{num_epochs}: Training Loss Pol: {avg_pol_train_loss}, Validation Loss Pol: {avg_pol_val_loss}')
    return training_aspect, training_polarization, validation_aspect, validation_polarization





def testing_run(model, batch_size, test_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    #print("this is the test dataloader: ",test_dataloader)

    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct_aspect_predictions = 0
    correct_polarization_predictions = 0

    
    for batch in test_dataloader:
        #print("coolness")
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        

        print(labels)
        
        with torch.no_grad():

                total_loss, aspect_loss, polarization_loss, aspect_logits, polarization_logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            
            # Calculate aspect accuracy
                aspect_predictions = torch.argmax(aspect_logits, dim=1)
                correct_aspect_predictions += torch.sum(aspect_predictions == labels[:, 0]).item()
           
    
            # Calculate polarization accuracy
                polarization_predictions = torch.argmax(polarization_logits, dim=1)
                correct_polarization_predictions += torch.sum(polarization_predictions == labels[:, 1]).item()

        total_samples += labels.size(0)

    average_loss = total_loss / len(test_dataset)
    aspect_accuracy = correct_aspect_predictions / total_samples
    polarization_accuracy = correct_polarization_predictions / total_samples

    print(f'Average Test Loss: {average_loss}')
    print(f'Aspect Accuracy: {aspect_accuracy}')
    print(f'Polarization Accuracy: {polarization_accuracy}')


_, test_dataset= setup_data(rests, tokenizer, 2970, 3714)
