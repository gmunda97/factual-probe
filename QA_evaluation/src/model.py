from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn


class QAWithTransformedEmbeddings(nn.Module):
    def __init__(self, transformation_matrix_path: str, input_dim: int = 768, output_dim: int = 2):
        super(QAWithTransformedEmbeddings, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.transformation_matrix = self.load_transformation_matrix(transformation_matrix_path, input_dim)
        self.qa_outputs = nn.Linear(input_dim, output_dim)

        for param in self.bert.parameters():
            param.requires_grad = False
    
    def load_transformation_matrix(self, path, input_dim):
        checkpoint = torch.load(path)
        transformation_matrix = checkpoint['model_class'](input_dim)
        transformation_matrix.load_state_dict(checkpoint['state_dict'])
        transformation_matrix.eval()
        return transformation_matrix

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        if self.transformation_matrix is not None:
            sequence_output = self.transformation_matrix(sequence_output)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)


transformation_matrix_path = './../../structural-probe/trained_models/42k_orthogonal_bert_cls.pth'
model = QAWithTransformedEmbeddings(transformation_matrix_path)