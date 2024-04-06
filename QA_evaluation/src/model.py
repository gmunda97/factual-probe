from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn


class QAWithTransformedEmbeddings(nn.Module):
    def __init__(self, transformation_matrix):
        super(QAWithTransformedEmbeddings, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.transformation_matrix = transformation_matrix
        self.qa_outputs = nn.Linear(768, 2) # Assuming 768 hidden size for BERT

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        transformed_embeddings = torch.matmul(sequence_output, self.transformation_matrix)
        logits = self.qa_outputs(transformed_embeddings)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)
