# IMPORTS
import numpy as np
import pandas as pd

import xml.etree.ElementTree as ET
from xml.dom import minidom

from transformers import BertModel, BertTokenizer, BertConfig, BertForSequenceClassification, AdamW, AutoTokenizer
from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification, RobertaConfig
from transformers import DebertaForSequenceClassification, DebertaConfig, DebertaTokenizer, get_scheduler
from transformers import TrainingArguments, Trainer, TrainerCallback

from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score
import evaluate
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import xml.etree.ElementTree as ET

# Define the class to encode dataset and function to slpit the dataset in train, val and test
# plus define function to encode training data

class ABSA_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def xml_to_df(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    data = []
    for sentence in root.findall('sentence'):
        sentence_id = sentence.get('id')
        text = sentence.find('text').text

        aspect_terms = sentence.find('aspectTerms')
        if aspect_terms is not None:
            for aspect_term in aspect_terms.findall('aspectTerm'):
                term = aspect_term.get('term')
                polarity = aspect_term.get('polarity')
                from_index = aspect_term.get('from')
                to_index = aspect_term.get('to')
                data.append([sentence_id, text, term, polarity, from_index, to_index])
        else:
            data.append([sentence_id, text, None, None, None, None])

    df = pd.DataFrame(data, columns=['sentence_id', 'text', 'term', 'polarity', 'from_index', 'to_index'])
    return df

#OR

def parse_data_2014(xml_file):
    container = []  # Initialize Container (List) for Parse Data
    sentences = ET.parse(xml_file).getroot()  # Get Sentence-Level Nodes

    for sentence in sentences:  # Loop Through Sentences
        sentence_id = sentence.attrib["id"]  # Save ID
        sentence_text = sentence.find('text').text  # Save Text
        aspects = sentence.findall('*')  # Get Aspect-Level Nodes

        found_category = False

        for aspect in aspects:  # Loop Through Aspects
            if aspect.tag == "aspectCategories":
                opinions = aspect.findall('*')  # Get Opinion-Level Nodes
                for opinion in opinions:
                    category = opinion.attrib["category"]
                    polarity = opinion.attrib.get("polarity", np.nan)
                    row = {"sentence_id": sentence_id, "sentence": sentence_text, "category": category, "polarity": polarity}
                    container.append(row)
                found_category = True

        if not found_category:
            row = {"sentence_id": sentence_id, "sentence": sentence_text, "category": np.nan, "polarity": np.nan}
            container.append(row)

    return pd.DataFrame(container)

# Functions for splitting the data and encode the input

def merge_sent(list1, list2):
    merged_ = []
    for x, y in zip(list1, list2):
        merged_.append(x + " [SEP] " + str(y))
    return merged_

def split_dataframe(df, train_percent=0.7, val_percent=0.15, test_percent=0.15):

    if train_percent + val_percent + test_percent != 1.0:
        raise ValueError("The sum of train_percent, val_percent, and test_percent must be equal to 1.0.")

    # Shuffle the DataFrame
    df = df.sample(frac=1).reset_index(drop=True)

    # Calculate split indices
    total_samples = len(df)
    train_end = int(train_percent * total_samples)
    val_end = train_end + int(val_percent * total_samples)

    # Split the DataFrame
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df

def setup_data(tokenizer, train_df, eval_df, test_df):

    label_map = {'negative': 0, 'neutral': 1, 'positive': 2, 'conflict': 3}
    label_map_category = {'service': 4, 'food': 5,'anecdotes/miscellaneous': 6,'price': 7, 'ambience': 8 }

    train_texts = train_df['sentence']
    train_labels = train_df['polarity']
    train_labels_category = train_df['category']
    train_labels = [label_map[label_] for label_ in train_labels]
    train_labels_category = [label_map_category[label] for label in train_labels_category]


    eval_texts = eval_df['sentence']
    eval_labels = eval_df['polarity']
    eval_labels_category = eval_df['category']
    eval_labels = [label_map[label_] for label_ in eval_labels]
    eval_labels_category = [label_map_category[label] for label in eval_labels_category]

    test_texts = test_df['sentence']
    test_labels = test_df['polarity']
    test_labels_category = test_df['category']
    test_labels = [label_map[label_] for label_ in test_labels]
    test_labels_category = [label_map_category[label] for label in test_labels_category]

    # Since DeBERTa does not accept 2 labels as input, I'm including the categories in the text embeddings
    # separated by a special tokens [SEP]
    train_ = merge_sent(train_texts, train_labels_category)
    eval_ = merge_sent(eval_texts, eval_labels_category)
    test_ = merge_sent(test_texts, test_labels_category)

    # train_encodings = tokenizer(str(train_texts), padding="max_length", truncation=True, max_length=512)
    # eval_encodings = tokenizer(str(eval_texts), padding="max_length", truncation=True, max_length=512)
    # test_encodings = tokenizer(str(test_texts), padding="max_length", truncation=True, max_length=512)

    train_encodings = tokenizer(train_, padding="max_length", truncation=True, max_length=512)
    eval_encodings = tokenizer(eval_, padding="max_length", truncation=True, max_length=512)
    test_encodings = tokenizer(test_, padding="max_length", truncation=True, max_length=512)

    #train_labels = [train_labels, train_labels_category]
    #eval_labels = [eval_labels, eval_labels_category]

    train_dataset = ABSA_Dataset(train_encodings, train_labels)
    val_dataset = ABSA_Dataset(eval_encodings, eval_labels)

    return train_dataset, val_dataset, test_
