import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, AdamW
from datasets import load_dataset
import matplotlib.pyplot as plt

# Define the model architecture
class CustomBERTForAspectPolarization(torch.nn.Module):
    def __init__(self, num_aspect_labels, num_polarization_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.aspect_classifier = torch.nn.Linear(self.bert.config.hidden_size, num_aspect_labels)
        self.polarization_classifier = torch.nn.Linear(self.bert.config.hidden_size, num_polarization_labels)
        self.dropout = torch.nn.Dropout(self.bert.config.hidden_dropout_prob)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = self.dropout(outputs.pooler_output)
        aspect_logits = self.aspect_classifier(pooled_output)
        polarization_logits = self.polarization_classifier(pooled_output)
        return aspect_logits, polarization_logits

# Load dataset
dataset = load_dataset("your_dataset_name")

# Tokenize dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encoded_dataset = dataset.map(lambda example: tokenizer(example['text'], padding=True, truncation=True), batched=True)

# Prepare DataLoader
train_loader = DataLoader(encoded_dataset['train'], batch_size=8, shuffle=True)

# Instantiate the model
num_aspect_labels = # number of aspect labels in your dataset - needs to be determined
num_polarization_labels = # number of polarization labels in your dataset
model = CustomBERTForAspectPolarization(num_aspect_labels, num_polarization_labels)

# Define optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
num_epochs = 3
losses = []
accuracies = []

for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch in train_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels_aspect = batch['aspect_labels']
        labels_polarization = batch['polarization_labels']
        
        optimizer.zero_grad()
        aspect_logits, polarization_logits = model(input_ids, attention_mask=attention_mask)
        
        aspect_loss = torch.nn.CrossEntropyLoss()(aspect_logits, labels_aspect)
        polarization_loss = torch.nn.CrossEntropyLoss()(polarization_logits, labels_polarization)
        loss = aspect_loss + polarization_loss
        
        loss.backward()
        optimizer.step()
        
        _, predicted_aspect = torch.max(aspect_logits, 1)
        _, predicted_polarization = torch.max(polarization_logits, 1)
        total += labels_aspect.size(0)
        correct += (predicted_aspect == labels_aspect).sum().item()
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct / total
    losses.append(epoch_loss)
    accuracies.append(epoch_accuracy)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

# Plotting loss and accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(accuracies)
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
