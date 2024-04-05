import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def training_run(model, output_dir, num_epochs, batch_size, train_dataset):
    epochs = num_epochs
    batch_size = 16
    num_steps = len(train_dataset) * epochs // batch_size
    warmup_steps = num_steps // 10  # 10% of the training steps
    save_steps = num_steps // epochs    # Save a checkpoint at the end of each epoch

    training_args = TrainingArguments(output_dir=output_dir,num_train_epochs = epochs,
          per_device_train_batch_size = batch_size,
    per_device_eval_batch_size = batch_size,
    warmup_steps = warmup_steps,
    weight_decay = 0.01,
    logging_dir = 'logs',
    logging_steps = 10,
    evaluation_strategy = 'epoch',
    learning_rate = 2e-5,
    save_steps = save_steps)
    
    trainer = Trainer(model,
    training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics = compute_metrics
)
    trainer.train() 
  


def parse_data_2014(xml_path):
    container = []  # Initialize Container (List) for Parse Data
    sentences = ET.parse(xml_path).getroot()  # Get Sentence-Level Nodes

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





def compute_metrics(eval_pred):
    """ evlaulations
    code to compute the metrics after taking the
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def setup_data(df):
    
    train_df = df.select([i for i in range(500)])
    eval_df = df.select([i for i in range(500,1000)])

    label_map = {'negative': 0, 'neutral': 1, 'positive': 2, 'conflict':3}  # Mapping of string labels to integer values




    train_texts = train_df['sentence']
    train_labels = train_df['polarity']
    train_labels =  [label_map[label] for label in train_labels]


    eval_texts = eval_df['sentence']
    eval_labels = eval_df['polarity']
    eval_labels =  [label_map[label] for label in eval_labels]



    train_encodings = tokenizer(train_texts, padding="max_length", truncation=True, max_length=512)
    eval_encodings = tokenizer(eval_texts, padding="max_length", truncation=True, max_length=512)

    train_dataset = ABSA_Dataset(train_encodings, train_labels)
    val_dataset = ABSA_Dataset(eval_encodings, eval_labels)

    return train_dataset, val_dataset



