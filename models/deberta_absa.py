classifier = nn.Sequential(nn.Dropout(), nn.Linear(768, 5), nn.Softmax())
classifer2 = nn.Sequential(nn.Dropout(), nn.Linear(768, 4), nn.Softmax())

class DebertaForABSA(nn.Module):
    def __init__(config, num_aspect_labels, num_polarization_labels)):
        super().__init__(config)
        self.deberta = DebertaModel(config)
        self.aspect_classifier = classifier
        self.polarization_classifier = classifer2
        #self.dropout = nn.Dropout(config.hidden_dropout_prob)

  def forward(self, input_ids, token_type_ids, attention_mask):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, 
                                     return_dict=False)
        
        logits1 = self.classifier1(pooled_output)
        logits2 = self.classifier2(pooled_output)

        return logits1, logits2
