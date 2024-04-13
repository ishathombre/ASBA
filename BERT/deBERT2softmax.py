from transformers import DebertaForSequenceClassification, DebertaModel
import torch
import torch.nn as nn

class CustomDebertaClassifier(DebertaForSequenceClassification):
    def __init__(self, config, num_aspect_labels, num_polarization_labels):
        super().__init__(config)
        # Add custom layers or modifications here
        self.bert = DebertaModel(config)
        self.aspect_classifier = nn.Linear(config.hidden_size, num_aspect_labels)
        self.polarization_classifier = nn.Linear(config.hidden_size, num_polarization_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds)
        pooled_output = self.dropout(outputs[0])
        aspect_logits = self.aspect_classifier(pooled_output)
        polarization_logits = self.polarization_classifier(pooled_output)

        outputs = (aspect_logits, polarization_logits,) + outputs[2:]
        print(outputs)  # Add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                aspect_loss_fct = nn.CrossEntropyLoss()
                polarization_loss_fct = nn.CrossEntropyLoss()
                aspect_loss = aspect_loss_fct(aspect_logits.view(-1, self.num_labels), labels[:, 0].view(-1))
                polarization_loss = polarization_loss_fct(polarization_logits.view(-1, self.num_labels), labels[:, 1].view(-1))
                loss = aspect_loss + polarization_loss
            outputs = (loss,) + outputs
        return outputs