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
